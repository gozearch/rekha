//! Per-collection inverted index over record metadata (Chroma's metadata
//! segment, simplified to one collection).
//!
//! Chroma's metadata segment maps `(metadata key, value) → bitmap of internal
//! offsets`. [`Postings`] is that structure, backed by [`roaring::RoaringBitmap`]
//! keyed by the per-collection internal `u32` offsets assigned by [`Collection`]
//! (monotonic, never reused — Chroma's offset scheme).
//!
//! # Derived state
//!
//! Postings are pure derived state. They are rebuilt by replaying the WAL on
//! reopen (the engine calls [`Postings::insert`]/[`Postings::remove`] from the
//! same `apply_operation` path used for live writes and replay), so a corrupted
//! or missing index is never a correctness problem.
//!
//! # Semantics contract
//!
//! [`Postings::evaluate`] must agree with [`WhereFilter::matches`]
//! (`rekha-core/src/filter.rs`) — that is the semantics oracle, and the
//! `filtered_query_*` integration tests assert set-equality between the two.
//! Two non-obvious consequences:
//!
//! - `$ne` requires the key to be **present** (absent keys never match a `$ne`
//!   in `WhereFilter::matches`), so `$ne` is computed as
//!   `(records carrying key) − eq[key][value]`, **not** `all − eq[key][value]`.
//! - `$nin` matches when the key is **absent or** not in the list, so it is
//!   `all − ∪ eq[key][v]` (key-absent records are in `all` and in no eq posting).
//!
//! Range scans (`$gt`/`$gte`/`$lt`/`$lte`) union the numeric postings in the
//! requested interval. This is O(interval size) per query; a production
//! implementation would keep a segment tree for logarithmic range queries. Fine
//! for this phase.

use std::collections::{BTreeMap, HashMap};
use std::ops::Bound;

use ordered_float::OrderedFloat;
use roaring::RoaringBitmap;

use rekha_core::filter::{ComparisonOp, WhereCondition, WhereFilter};
use rekha_core::types::{Metadata, MetadataValue};

/// Canonical, comparable form of a metadata value. Numbers normalize int/float
/// so `Int(5)` and `Float(5.0)` collide (Chroma's numeric coercion).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ValueKey {
    Str(String),
    Bool(bool),
    Num(OrderedFloat<f64>),
}

impl ValueKey {
    /// Canonicalize a metadata value into its comparable form.
    pub fn from_value(v: &MetadataValue) -> Self {
        match v {
            MetadataValue::Str(s) => ValueKey::Str(s.clone()),
            MetadataValue::Bool(b) => ValueKey::Bool(*b),
            MetadataValue::Int(i) => ValueKey::Num(OrderedFloat(*i as f64)),
            MetadataValue::Float(f) => ValueKey::Num(OrderedFloat(*f)),
        }
    }
}

/// Inverted index: for every `(key, value)` pair, the set of live offsets whose
/// record metadata contains it. Rebuilt from the WAL on reopen (derived state).
pub struct Postings {
    /// Exact-match postings: key → value → offsets. Numeric values also appear
    /// here so `$eq`/`$ne`/`$in`/`$nin` work with int/float coercion.
    eq: HashMap<String, HashMap<ValueKey, RoaringBitmap>>,
    /// Numeric postings: key → sorted value → offsets, for range scans
    /// (`$gt`/`$gte`/`$lt`/`$lte`).
    num: HashMap<String, BTreeMap<OrderedFloat<f64>, RoaringBitmap>>,
    /// All live offsets (used for `$ne`/`$nin` which match against the
    /// complement).
    all: RoaringBitmap,
}

impl Default for Postings {
    fn default() -> Self {
        Self::new()
    }
}

impl Postings {
    pub fn new() -> Self {
        Self {
            eq: HashMap::new(),
            num: HashMap::new(),
            all: RoaringBitmap::new(),
        }
    }

    /// Index `metadata` for `offset`. Call with the NEW metadata after a write;
    /// the caller must first remove the old metadata if it changed. Records
    /// with no metadata are still tracked in `all` (a `$nin` matches them), so
    /// an `Option<&Metadata>` is accepted.
    pub fn insert(&mut self, offset: u32, metadata: Option<&Metadata>) {
        self.all.insert(offset);
        let Some(metadata) = metadata else {
            return;
        };
        for (key, value) in metadata {
            let vk = ValueKey::from_value(value);
            self.eq
                .entry(key.clone())
                .or_default()
                .entry(vk)
                .or_default()
                .insert(offset);
            if let Some(n) = value.as_f64() {
                self.num
                    .entry(key.clone())
                    .or_default()
                    .entry(OrderedFloat(n))
                    .or_default()
                    .insert(offset);
            }
        }
    }

    /// Un-index `metadata` for `offset`. Call with the OLD metadata before
    /// overwriting or on delete. Empty per-key bitmaps are pruned.
    pub fn remove(&mut self, offset: u32, metadata: Option<&Metadata>) {
        self.all.remove(offset);
        let Some(metadata) = metadata else {
            return;
        };
        for (key, value) in metadata {
            let vk = ValueKey::from_value(value);
            if let Some(key_map) = self.eq.get_mut(key) {
                if let Some(bm) = key_map.get_mut(&vk) {
                    bm.remove(offset);
                    if bm.is_empty() {
                        key_map.remove(&vk);
                    }
                }
                if key_map.is_empty() {
                    self.eq.remove(key);
                }
            }
            if let Some(n) = value.as_f64() {
                if let Some(key_map) = self.num.get_mut(key) {
                    let of = OrderedFloat(n);
                    if let Some(bm) = key_map.get_mut(&of) {
                        bm.remove(offset);
                        if bm.is_empty() {
                            key_map.remove(&of);
                        }
                    }
                    if key_map.is_empty() {
                        self.num.remove(key);
                    }
                }
            }
        }
    }

    /// Drop `offset` from every index and from `all` (delete path).
    pub fn remove_all(&mut self, offset: u32) {
        for key_map in self.eq.values_mut() {
            for bm in key_map.values_mut() {
                bm.remove(offset);
            }
            key_map.retain(|_, bm| !bm.is_empty());
        }
        self.eq.retain(|_, key_map| !key_map.is_empty());
        for key_map in self.num.values_mut() {
            for bm in key_map.values_mut() {
                bm.remove(offset);
            }
            key_map.retain(|_, bm| !bm.is_empty());
        }
        self.num.retain(|_, key_map| !key_map.is_empty());
        self.all.remove(offset);
    }

    /// Bitmap of offsets matching the filter. `And` intersects, `Or` unions,
    /// comparisons resolve as:
    ///
    /// - `Eq(v)` → `eq[key][ValueKey::from_value(v)]` (empty if absent)
    /// - `Ne(v)` → `(records carrying key) − eq[key][v]` (absent keys never
    ///   match, mirroring `WhereFilter::matches`)
    /// - `Gt/Gte/Lt/Lte(n)` → union of `num[key]` range (empty if key absent)
    /// - `In(vs)` → union of `eq[key][ValueKey::from_value(v)]` for each `v`
    /// - `Nin(vs)` → `all − ∪ eq[key][v]` (matches records that lack the key
    ///   too — mirror `WhereFilter::matches` semantics)
    ///
    /// Absent keys/values → empty bitmap (except `Ne`/`Nin` which use `all`).
    pub fn evaluate(&self, filter: &WhereFilter) -> RoaringBitmap {
        eval_condition(self, &filter.condition)
    }

    /// Live count, equal to `all.len()`.
    pub fn len(&self) -> usize {
        self.all.len() as usize
    }

    pub fn is_empty(&self) -> bool {
        self.all.is_empty()
    }

    /// Bitmap of offsets carrying `key` (any value). `$ne` needs it because it
    /// only matches records where the key is present.
    fn key_present(&self, key: &str) -> RoaringBitmap {
        match self.eq.get(key) {
            Some(key_map) => key_map
                .values()
                .fold(RoaringBitmap::new(), |acc, bm| acc | bm),
            None => RoaringBitmap::new(),
        }
    }

    /// Exact-match postings for `(key, value)`, or an empty bitmap.
    fn eq_bitmap(&self, key: &str, v: &ValueKey) -> RoaringBitmap {
        self.eq
            .get(key)
            .and_then(|key_map| key_map.get(v))
            .cloned()
            .unwrap_or_default()
    }

    /// Union of numeric postings in `[lo, hi]`, or an empty bitmap.
    fn num_range(
        &self,
        key: &str,
        lo: Bound<OrderedFloat<f64>>,
        hi: Bound<OrderedFloat<f64>>,
    ) -> RoaringBitmap {
        match self.num.get(key) {
            Some(key_map) => key_map
                .range((lo, hi))
                .fold(RoaringBitmap::new(), |acc, (_, bm)| acc | bm),
            None => RoaringBitmap::new(),
        }
    }
}

fn eval_condition(postings: &Postings, cond: &WhereCondition) -> RoaringBitmap {
    match cond {
        WhereCondition::Comparison { key, op } => eval_op(postings, key, op),
        WhereCondition::And(conds) => {
            let Some((first, rest)) = conds.split_first() else {
                // The parser rejects empty lists; a defensive empty `$and`
                // matches everything.
                return postings.all.clone();
            };
            let mut out = eval_condition(postings, first);
            for c in rest {
                out &= eval_condition(postings, c);
            }
            out
        }
        WhereCondition::Or(conds) => {
            let mut out = RoaringBitmap::new();
            for c in conds {
                out |= eval_condition(postings, c);
            }
            out
        }
    }
}

fn eval_op(postings: &Postings, key: &str, op: &ComparisonOp) -> RoaringBitmap {
    match op {
        ComparisonOp::Eq(v) => postings.eq_bitmap(key, &ValueKey::from_value(v)),
        ComparisonOp::Ne(v) => {
            let present = postings.key_present(key);
            present - &postings.eq_bitmap(key, &ValueKey::from_value(v))
        }
        ComparisonOp::Gt(n) => {
            postings.num_range(key, Bound::Excluded(OrderedFloat(*n)), Bound::Unbounded)
        }
        ComparisonOp::Gte(n) => {
            postings.num_range(key, Bound::Included(OrderedFloat(*n)), Bound::Unbounded)
        }
        ComparisonOp::Lt(n) => {
            postings.num_range(key, Bound::Unbounded, Bound::Excluded(OrderedFloat(*n)))
        }
        ComparisonOp::Lte(n) => {
            postings.num_range(key, Bound::Unbounded, Bound::Included(OrderedFloat(*n)))
        }
        ComparisonOp::In(vs) => {
            let mut out = RoaringBitmap::new();
            for v in vs {
                out |= postings.eq_bitmap(key, &ValueKey::from_value(v));
            }
            out
        }
        ComparisonOp::Nin(vs) => {
            let mut excluded = RoaringBitmap::new();
            for v in vs {
                excluded |= postings.eq_bitmap(key, &ValueKey::from_value(v));
            }
            &postings.all - &excluded
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta(pairs: &[(&str, MetadataValue)]) -> Metadata {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect()
    }

    fn offsets(bm: &RoaringBitmap) -> Vec<u32> {
        bm.iter().collect()
    }

    fn parse(s: &str) -> WhereFilter {
        WhereFilter::parse_json(s).unwrap()
    }

    #[test]
    fn insert_remove_evaluate_eq() {
        let mut p = Postings::new();
        let a = meta(&[("tag", MetadataValue::Str("a".into()))]);
        let b = meta(&[("tag", MetadataValue::Str("b".into()))]);
        p.insert(1, Some(&a));
        p.insert(2, Some(&b));
        p.insert(3, None);
        assert_eq!(p.len(), 3);

        assert_eq!(
            offsets(&p.evaluate(&parse(r#"{"tag": {"$eq": "a"}}"#))),
            vec![1]
        );
        assert_eq!(
            offsets(&p.evaluate(&parse(r#"{"tag": {"$eq": "b"}}"#))),
            vec![2]
        );
        assert!(p.evaluate(&parse(r#"{"tag": {"$eq": "c"}}"#)).is_empty());

        // Overwrite record 1: tag a -> b.
        p.remove(1, Some(&a));
        p.insert(1, Some(&b));
        assert!(p.evaluate(&parse(r#"{"tag": {"$eq": "a"}}"#)).is_empty());
        assert_eq!(
            offsets(&p.evaluate(&parse(r#"{"tag": {"$eq": "b"}}"#))),
            vec![1, 2]
        );
    }

    #[test]
    fn numeric_coercion_eq() {
        let mut p = Postings::new();
        let int5 = meta(&[("score", MetadataValue::Int(5))]);
        let flt5 = meta(&[("score", MetadataValue::Float(5.0))]);
        let six = meta(&[("score", MetadataValue::Int(6))]);
        p.insert(1, Some(&int5));
        p.insert(2, Some(&flt5));
        p.insert(3, Some(&six));

        // Int(5) and Float(5.0) collide on the ValueKey::Num(5.0) posting.
        assert_eq!(
            offsets(&p.evaluate(&parse(r#"{"score": {"$eq": 5}}"#))),
            vec![1, 2]
        );
        assert_eq!(
            offsets(&p.evaluate(&parse(r#"{"score": {"$eq": 5.0}}"#))),
            vec![1, 2]
        );
        assert_eq!(
            offsets(&p.evaluate(&parse(r#"{"score": {"$eq": 6}}"#))),
            vec![3]
        );
    }

    #[test]
    fn ne_requires_key_present() {
        let mut p = Postings::new();
        p.insert(1, Some(&meta(&[("tag", MetadataValue::Str("a".into()))])));
        p.insert(2, Some(&meta(&[("tag", MetadataValue::Str("b".into()))])));
        p.insert(3, None);
        p.insert(4, Some(&meta(&[("other", MetadataValue::Int(1))])));

        // $ne matches key-present != a; key-absent records (3, 4) never match.
        let got = p.evaluate(&parse(r#"{"tag": {"$ne": "a"}}"#));
        assert_eq!(offsets(&got), vec![2]);
    }

    #[test]
    fn nin_matches_absent_key() {
        let mut p = Postings::new();
        p.insert(1, Some(&meta(&[("tag", MetadataValue::Str("a".into()))])));
        p.insert(2, Some(&meta(&[("tag", MetadataValue::Str("b".into()))])));
        p.insert(3, None);
        p.insert(4, Some(&meta(&[("other", MetadataValue::Int(1))])));
        p.insert(5, Some(&meta(&[("tag", MetadataValue::Str("c".into()))])));

        assert_eq!(
            offsets(&p.evaluate(&parse(r#"{"tag": {"$nin": ["a"]}}"#))),
            vec![2, 3, 4, 5]
        );
        // Absent key OR value not in list.
        assert_eq!(
            offsets(&p.evaluate(&parse(r#"{"tag": {"$nin": ["a", "c"]}}"#))),
            vec![2, 3, 4]
        );
    }

    #[test]
    fn in_union() {
        let mut p = Postings::new();
        p.insert(1, Some(&meta(&[("tag", MetadataValue::Str("a".into()))])));
        p.insert(2, Some(&meta(&[("tag", MetadataValue::Str("b".into()))])));
        p.insert(3, Some(&meta(&[("tag", MetadataValue::Str("c".into()))])));
        assert_eq!(
            offsets(&p.evaluate(&parse(r#"{"tag": {"$in": ["a", "b"]}}"#))),
            vec![1, 2]
        );
    }

    #[test]
    fn range_boundaries() {
        let mut p = Postings::new();
        for (off, score) in [(1, 10.0f64), (2, 50.0), (3, 51.0), (4, 99.0)] {
            p.insert(off, Some(&meta(&[("score", MetadataValue::Float(score))])));
        }
        // Non-numeric value must not appear in numeric ranges.
        p.insert(5, Some(&meta(&[("score", MetadataValue::Str("x".into()))])));

        assert_eq!(
            offsets(&p.evaluate(&parse(r#"{"score": {"$gt": 50}}"#))),
            vec![3, 4]
        );
        assert_eq!(
            offsets(&p.evaluate(&parse(r#"{"score": {"$gte": 50}}"#))),
            vec![2, 3, 4]
        );
        assert_eq!(
            offsets(&p.evaluate(&parse(r#"{"score": {"$lt": 50}}"#))),
            vec![1]
        );
        assert_eq!(
            offsets(&p.evaluate(&parse(r#"{"score": {"$lte": 50}}"#))),
            vec![1, 2]
        );
        // Absent key -> empty for ranges.
        assert!(p.evaluate(&parse(r#"{"nope": {"$gt": 0}}"#)).is_empty());
    }

    #[test]
    fn and_or_nesting() {
        let mut p = Postings::new();
        p.insert(
            1,
            Some(&meta(&[
                ("tag", MetadataValue::Str("a".into())),
                ("score", MetadataValue::Int(80)),
            ])),
        );
        p.insert(
            2,
            Some(&meta(&[
                ("tag", MetadataValue::Str("a".into())),
                ("score", MetadataValue::Int(5)),
            ])),
        );
        p.insert(
            3,
            Some(&meta(&[
                ("tag", MetadataValue::Str("b".into())),
                ("score", MetadataValue::Int(90)),
            ])),
        );

        assert_eq!(
            offsets(&p.evaluate(&parse(
                r#"{"$and": [{"tag": "a"}, {"score": {"$gte": 50}}]}"#
            ))),
            vec![1]
        );
        assert_eq!(
            offsets(&p.evaluate(&parse(r#"{"$or": [{"tag": "a"}, {"score": {"$lt": 50}}]}"#))),
            vec![1, 2]
        );
        // Nesting: tag a AND (score >= 50 OR score < 10) => both a records.
        assert_eq!(
            offsets(&p.evaluate(&parse(
                r#"{"$and": [{"tag": "a"}, {"$or": [{"score": {"$gte": 50}}, {"score": {"$lt": 10}}]}]}"#
            ))),
            vec![1, 2]
        );
    }

    #[test]
    fn delete_clears_all_indexes() {
        let mut p = Postings::new();
        p.insert(
            1,
            Some(&meta(&[
                ("tag", MetadataValue::Str("a".into())),
                ("score", MetadataValue::Int(5)),
            ])),
        );
        p.insert(2, Some(&meta(&[("tag", MetadataValue::Str("a".into()))])));
        assert_eq!(p.len(), 2);

        p.remove(
            1,
            Some(&meta(&[
                ("tag", MetadataValue::Str("a".into())),
                ("score", MetadataValue::Int(5)),
            ])),
        );
        p.remove_all(1);

        assert_eq!(p.len(), 1);
        let tag_a = p.evaluate(&parse(r#"{"tag": {"$eq": "a"}}"#));
        assert_eq!(offsets(&tag_a), vec![2]);
        assert!(p.evaluate(&parse(r#"{"score": {"$gte": 0}}"#)).is_empty());
        // Record 2 still eligible for $nin (key tag present with value a).
        assert!(
            p.evaluate(&parse(r#"{"tag": {"$nin": ["b"]}}"#))
                .contains(2)
        );
    }

    #[test]
    fn remove_all_even_without_prior_remove() {
        let mut p = Postings::new();
        p.insert(1, Some(&meta(&[("tag", MetadataValue::Str("a".into()))])));
        p.remove_all(1);
        assert_eq!(p.len(), 0);
        assert!(p.evaluate(&parse(r#"{"tag": {"$eq": "a"}}"#)).is_empty());
        assert!(p.evaluate(&parse(r#"{"tag": {"$nin": ["b"]}}"#)).is_empty());
    }

    #[test]
    fn matches_where_filter_oracle() {
        // Cross-check a few arbitrary filters against WhereFilter::matches on
        // the same metadata, on a small corpus.
        let corpus = [
            (1u32, meta(&[("tag", MetadataValue::Str("a".into()))])),
            (
                2,
                meta(&[
                    ("tag", MetadataValue::Str("b".into())),
                    ("score", MetadataValue::Int(30)),
                ]),
            ),
            (
                3,
                meta(&[
                    ("tag", MetadataValue::Str("a".into())),
                    ("score", MetadataValue::Float(70.0)),
                ]),
            ),
            (4, meta(&[("score", MetadataValue::Int(5))])),
        ];
        let mut p = Postings::new();
        for (off, md) in &corpus {
            p.insert(*off, Some(md));
        }
        let filters = [
            r#"{"tag": {"$eq": "a"}}"#,
            r#"{"tag": {"$ne": "a"}}"#,
            r#"{"tag": {"$in": ["a", "b"]}}"#,
            r#"{"tag": {"$nin": ["a"]}}"#,
            r#"{"score": {"$gt": 20}}"#,
            r#"{"score": {"$lte": 30}}"#,
            r#"{"$and": [{"tag": "a"}, {"score": {"$gte": 70}}]}"#,
            r#"{"$or": [{"score": {"$lt": 10}}, {"tag": "b"}]}"#,
        ];
        for f in filters {
            let filter = parse(f);
            let expected: Vec<u32> = corpus
                .iter()
                .filter(|(_, md)| filter.matches(md))
                .map(|(off, _)| *off)
                .collect();
            assert_eq!(offsets(&p.evaluate(&filter)), expected, "filter {f}");
        }
    }
}
