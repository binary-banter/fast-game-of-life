#[cfg(feature = "cuda")]
use fast_game_of_life::cuda::Game;
#[cfg(feature = "opencl")]
use fast_game_of_life::opencl::Game;
#[cfg(feature = "simd")]
use fast_game_of_life::simd::Game;
#[cfg(not(any(feature = "simd", feature = "cuda", feature = "opencl")))]
use fast_game_of_life::trivial::Game;

use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn step_1024(c: &mut Criterion) {
    let mut game: Game = Game::new(65536, 65536);
    c.bench_function("step_1024", |b| b.iter(|| black_box(&mut game).step(1024)));
}

criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = step_1024
);
criterion_main!(benches);
