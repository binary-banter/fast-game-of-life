use crate::cuda::driver::args::Args;
use crate::cuda::driver::check_error;
use crate::cuda::driver::kernel::Kernel;
use cuda_runtime_sys::{
    cudaError, cudaLaunchKernel, cudaStreamCreate, cudaStreamDestroy, cudaStreamSynchronize,
    cudaStream_t, dim3,
};
use std::ptr;

pub struct Stream {
    pointer: cudaStream_t,
}

impl Stream {
    pub fn new() -> Result<Self, cudaError> {
        let mut pointer: cudaStream_t = ptr::null_mut();
        unsafe { check_error(cudaStreamCreate(&mut pointer as *mut cudaStream_t))? }
        Ok(Self { pointer })
    }

    pub fn launch(
        &mut self,
        kernel: Kernel,
        grid_dim: dim3,
        block_dim: dim3,
        args: &Args,
        shared_mem: usize,
    ) -> Result<(), cudaError> {
        unsafe {
            check_error(cudaLaunchKernel(
                kernel.function(),
                grid_dim,
                block_dim,
                args.as_args(),
                shared_mem,
                self.pointer,
            ))
        }
    }

    pub fn wait(&self) -> Result<(), cudaError> {
        unsafe { check_error(cudaStreamSynchronize(self.pointer)) }
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        unsafe {
            check_error(cudaStreamDestroy(self.pointer)).unwrap();
        }
    }
}
