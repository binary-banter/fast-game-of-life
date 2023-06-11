#[cfg(not(feature = "cuda"))]
mod opencl;

use cuda_runtime_sys::{dim3};
#[cfg(not(feature = "cuda"))]
use opencl as game;
use std::os::raw::c_void;

#[cfg(feature = "cuda")]
mod cuda;

use crate::cuda::driver::args::Args;
use crate::cuda::driver::buffer::Buffer;
use crate::cuda::driver::kernel::Kernel;
use crate::cuda::driver::stream::Stream;
#[cfg(feature = "cuda")]
use cuda as game;

#[link(name = "kernel", kind = "static")]
extern "C" {
    fn step() -> c_void;
}

pub fn main() {
    let grid_dim = dim3 { x: 64, y: 1, z: 1 };
    let block_dim = dim3 { x: 1, y: 1, z: 1 };

    let mut buffer = Buffer::new(64).unwrap();
    buffer.write_all(0).unwrap();

    let kernel = Kernel::new(step);
    let mut args = Args::new();
    args.add_arg(&mut buffer);

    let mut stream = Stream::new().unwrap();
    stream
        .launch(kernel, grid_dim, block_dim, &args, 0)
        .unwrap();

    assert!(buffer.read_all().unwrap().into_iter().all(|a| a == 1));
}
