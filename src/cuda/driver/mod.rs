pub mod args;
pub mod buffer;
pub mod kernel;
pub mod stream;

use cuda_runtime_sys::cudaError;

pub fn check_error(error: cudaError) -> Result<(), cudaError> {
    match error {
        cudaError::cudaSuccess => Ok(()),
        _ => Err(error),
    }
}
