# Unsafe code audit

`unsafe_code` is **denied** workspace-wide (`[workspace.lints.rust]` in
`Cargo.toml`), and `clippy::undocumented_unsafe_blocks` is denied in
`rekha-engine`. Both are enforced in CI, so this page cannot drift from the code
by omission: adding an `unsafe` block fails the build until it carries a
`// SAFETY:` comment *and* a per-site `#[allow(unsafe_code)]`.

There is exactly **one** `unsafe` block in the workspace.

## The block

`crates/rekha-engine/src/segment.rs`, `Segment::from_file`:

```rust
let mmap = unsafe { memmap2::Mmap::map(&file)? };
```

### Why it is unsafe

`Mmap::map` produces a `&[u8]` over a file. Rust's aliasing rules assume that a
shared reference's bytes do not change. A concurrent write or truncate through
another handle would violate that, and the result is undefined behaviour rather
than a wrong answer.

### What holds the invariant

1. **Segment files are write-once.** A segment is created inside a checkpoint
   generation directory and is never modified after publication. Compaction
   writes a *new* generation; it does not edit an existing segment.
2. **Garbage collection unlinks, it does not rewrite.** Removing an old
   generation decrements a link count. Existing mappings remain valid and their
   contents are unchanged, which is exactly the semantics `Mmap` requires.
3. **The file handle outlives the mapping.** `from_file` moves the `File` into
   the `Segment` as `_file`, so the inode cannot be recycled while the mapping
   exists. That field is not dead weight: it *is* half the safety argument, which
   is why it is named with a leading underscore rather than removed.

### What it does not protect against

An operator truncating or overwriting a segment file underneath a running node.
That is outside the program's control in the same way that editing a heap file
under a running database is. The supported check is `rekha verify` against a
stopped data directory.

## Why not `cargo miri`

`unsafe-miri-ci` would normally apply to any crate containing `unsafe`. It cannot
apply here: the block maps a file with `mmap`, which Miri does not model, and the
segment tests perform real `fs` I/O that Miri cannot execute. Running Miri over
these tests is not possible, so the assurance is the invariant above plus the
lint that forces it to be written down.

If a future `unsafe` block is *not* I/O-bound, it should be covered by Miri and
this page should say so.

## Review checklist for a new `unsafe` block

- Is there a safe alternative? Prefer it even at a performance cost.
- Is the block as small as possible? Only the operation that needs unsafety.
- Does `// SAFETY:` state the invariant, and what would break it?
- Is the invariant enforced by *structure* (ownership, immutability) or only by
  convention? Structure survives refactoring; convention does not.
- Add an entry here in the same commit.
