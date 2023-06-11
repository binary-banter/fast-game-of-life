use criterion::{black_box, criterion_group, criterion_main, Criterion};
use gol_cuda::game::Game;

fn step_1000(c: &mut Criterion) {
    //1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144,
    let mut game = Game::new(65536, 65536);
    c.bench_function("step_1000", |b| b.iter(|| black_box(&mut game).step(1024)));
}

criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = step_1000
);
criterion_main!(benches);
