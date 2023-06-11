#[cfg(not(feature = "cuda"))]
mod opencl;

use cuda_runtime_sys::{cudaLaunchKernel, cudaStreamCreate, cudaStream_t, dim3};
#[cfg(not(feature = "cuda"))]
use opencl as game;
use std::mem::size_of;
use std::os::raw::c_void;
use std::{ptr, slice};

#[cfg(feature = "cuda")]
mod cuda;

use crate::cuda::driver::check_error;
#[cfg(feature = "cuda")]
use cuda as game;

#[link(name = "kernel", kind = "static")]
extern "C" {
    fn step() -> c_void;
}

pub fn main() {
    let grid_dim = dim3 { x: 1, y: 1, z: 1 };

    let block_dim = dim3 { x: 1, y: 1, z: 1 };

    let mut stream: cudaStream_t = ptr::null_mut();
    unsafe {
        check_error(cudaStreamCreate(&mut stream as *mut cudaStream_t)).unwrap();
        check_error(cudaLaunchKernel(
            step as *const c_void,
            grid_dim,
            block_dim,
            ptr::null_mut(),
            0,
            stream,
        ))
        .unwrap();
    }
}
