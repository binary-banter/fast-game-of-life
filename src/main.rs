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
    let mut game = Game::new(32, 32);

    // blinker
    game.set(1, 0);
    game.set(1, 1);
    game.set(1, 2);

    game.step(1);

    println!("{game}");
}
