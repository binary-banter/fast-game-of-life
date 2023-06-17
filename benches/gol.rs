use criterion::{black_box, criterion_group, criterion_main, Criterion};
use fast_game_of_life::game::Game;

fn step_1024(c: &mut Criterion) {
    //1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144,
    let mut game: Game = Game::new(4096, 4096);
    c.bench_function("step_1024", |b| b.iter(|| black_box(&mut game).step(64)));
}

criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = step_1024
);
criterion_main!(benches);
