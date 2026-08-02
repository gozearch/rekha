use criterion::{criterion_group, criterion_main, Criterion};
use rand::Rng;
use rekha_quant::ProductQuantizer;

fn bench_pq_encode(c: &mut Criterion) {
    let dim = 128;
    let mut pq = ProductQuantizer::new(16, 64, dim).unwrap();

    let mut rng = rand::thread_rng();
    let vectors: Vec<Vec<f32>> = (0..500)
        .map(|_| (0..dim).map(|_| rng.gen::<f32>()).collect())
        .collect();
    let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
    pq.train(&refs).unwrap();

    let query: Vec<f32> = (0..dim).map(|_| rng.gen()).collect();

    c.bench_function("pq_encode_128d", |b| {
        b.iter(|| {
            pq.encode(&query).unwrap();
        });
    });

    c.bench_function("pq_distance_table_128d", |b| {
        b.iter(|| {
            pq.distance_table(&query);
        });
    });
}

criterion_group!(benches, bench_pq_encode);
criterion_main!(benches);
