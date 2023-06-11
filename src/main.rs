#[cfg(not(feature = "cuda"))]
mod opencl;

#[cfg(not(feature = "cuda"))]
use opencl as game;

#[cfg(feature = "cuda")]
mod cuda;

#[cfg(feature = "cuda")]
use cuda as game;

use game::Game;

pub fn main() {
    let mut game = Game::new(1024, 1024);
    const STEPS: usize = 1000;
    game.set(4, 5);
    game.set(5, 6);
    game.set(6, 4);
    game.set(6, 5);
    game.set(6, 6);
    game.step(4 * STEPS as u32);
    assert!(game.get(4 + STEPS, 5 + STEPS));
    assert!(game.get(5 + STEPS, 6 + STEPS));
    assert!(game.get(6 + STEPS, 4 + STEPS));
    assert!(game.get(6 + STEPS, 5 + STEPS));
    assert!(game.get(6 + STEPS, 6 + STEPS));
}
