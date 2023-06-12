#[cfg(not(any(feature = "opencl", feature = "cuda", feature = "simd")))]
compile_error!("No features were selected!");

#[cfg(all(feature = "opencl", feature = "cuda"))]
compile_error!("The \"opencl\" and \"cuda\" features are mutually exclusive!");

#[cfg(all(feature = "opencl", feature = "simd"))]
compile_error!("The \"opencl\" and \"simd\" features are mutually exclusive!");

#[cfg(all(feature = "cuda", feature = "simd"))]
compile_error!("The \"cuda\" and \"simd\" features are mutually exclusive!");

#[cfg(feature = "opencl")]
mod opencl;
#[cfg(feature = "opencl")]
use opencl as game;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
use cuda as game;

#[cfg(feature = "simd")]
pub mod simd;
#[cfg(feature = "simd")]
use simd as game;

use crate::game::Game;

pub fn main() {
    let mut game = Game::new(16, 16);
    game.set(4, 5);
    game.set(5, 6);
    game.set(6, 4);
    game.set(6, 5);
    game.set(6, 6);
    game.step(1);
    println!("{}", game);
}
