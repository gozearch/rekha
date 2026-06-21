use criterion::{criterion_group, criterion_main, Criterion};
use rand::Rng;
use rekha_index::{IvfIndex, ProductQuantizer};

fn bench_pq_encode(c: &mut Criterion) {
    let dim = 768;
    let mut pq = ProductQuantizer::new(64, 256, dim).unwrap();

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

fn bench_ivf_search(c: &mut Criterion) {
    let dim = 128;
    let mut rng = rand::thread_rng();
    let n = 5000;

    let vectors: Vec<(u64, Vec<f32>)> = (0..n)
        .map(|i| {
            let v: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();
            (i as u64, v)
        })
        .collect();

    let ivf = IvfIndex::build(&vectors, 32, 8, 4, 16, dim).unwrap();
    let query: Vec<f32> = (0..dim).map(|_| rng.gen()).collect();

    c.bench_function("ivf_search_k10", |b| {
        b.iter(|| {
            ivf.search(&query, 10, Some(4)).unwrap();
        });
    });

    c.bench_function("ivf_search_k10_nprobe8", |b| {
        b.iter(|| {
            ivf.search(&query, 10, Some(8)).unwrap();
        });
    });
}

criterion_group!(benches, bench_pq_encode, bench_ivf_search);
criterion_main!(benches);
