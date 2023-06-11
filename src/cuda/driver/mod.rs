mod buffer;
mod kernel;

use cuda_runtime_sys::cudaError;

pub fn check_error(error: cudaError) -> Result<(), cudaError> {
    match error {
        cudaError::cudaSuccess => Ok(()),
        _ => Err(error),
    }
}
