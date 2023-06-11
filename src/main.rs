#[cfg(not(feature = "cuda"))]
mod opencl;

use cuda_runtime_sys::dim3;
#[cfg(not(feature = "cuda"))]
use opencl as game;
use std::os::raw::c_void;
use std::thread::sleep;
use std::time::Duration;

#[cfg(feature = "cuda")]
mod cuda;

use crate::cuda::driver::args::Args;
use crate::cuda::driver::buffer::Buffer;
use crate::cuda::driver::kernel::Kernel;
use crate::cuda::driver::stream::Stream;
#[cfg(feature = "cuda")]
use cuda as game;
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
