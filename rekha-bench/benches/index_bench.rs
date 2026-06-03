use criterion::{criterion_group, criterion_main, Criterion};
use rand::Rng;
use rekha_core::SearchParams;
use rekha_index::{ProductQuantizer, VamanaGraph};

fn bench_pq_encode(c: &mut Criterion) {
    let dim = 768;
    let mut pq = ProductQuantizer::new(64, 256, dim).unwrap();

    // Generate training data.
    let mut rng = rand::thread_rng();
    let vectors: Vec<Vec<f32>> = (0..1000)
        .map(|_| (0..dim).map(|_| rng.gen::<f32>()).collect())
        .collect();
    let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
    pq.train(&refs).unwrap();

    let query: Vec<f32> = (0..dim).map(|_| rng.gen()).collect();

    c.bench_function("pq_encode", |b| {
        b.iter(|| {
            pq.encode(&query).unwrap();
        });
    });

    c.bench_function("pq_distance_table", |b| {
        b.iter(|| {
            pq.distance_table(&query);
        });
    });
}

fn bench_vamana_graph(c: &mut Criterion) {
    let dim = 128;
    let mut rng = rand::thread_rng();
    let n = 1000;

    let vectors: Vec<Vec<f32>> = (0..n)
        .map(|_| (0..dim).map(|_| rng.gen::<f32>()).collect())
        .collect();

    let refs: Vec<(u64, &[f32])> = vectors
        .iter()
        .enumerate()
        .map(|(i, v)| (i as u64, v.as_slice()))
        .collect();

    let mut graph = VamanaGraph::new(32);
    graph.build(&refs).unwrap();

    let query: Vec<f32> = (0..dim).map(|_| rng.gen()).collect();
    let vec_owned: Vec<(u64, Vec<f32>)> = vectors
        .iter()
        .enumerate()
        .map(|(i, v)| (i as u64, v.clone()))
        .collect();

    c.bench_function("vamana_search_k10", |b| {
        b.iter(|| {
            graph.search(&query, &vec_owned, 10, 64).unwrap();
        });
    });
}

criterion_group!(benches, bench_pq_encode, bench_vamana_graph);
criterion_main!(benches);
