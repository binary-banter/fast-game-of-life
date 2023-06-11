use crate::cuda::driver::check_error;
use cuda_runtime_sys::{cudaError, cudaFree, cudaMalloc, cudaMemcpy, cudaMemcpyKind};
use std::os::raw::c_void;
use std::ptr;

pub struct Buffer {
    pointer: *mut c_void,
    bytes: usize,
}

impl Buffer {
    pub fn new(bytes: usize) -> Result<Self, cudaError> {
        let mut pointer: *mut c_void = ptr::null_mut();
        unsafe {
            check_error(cudaMalloc(&mut pointer as *mut _, bytes))?;
        }

        Ok(Self { pointer, bytes })
    }

    /// reads bytes length of bytes from buffer using an offset
    /// panics if it offset + bytes length overflows the buffer size
    pub fn get(&self, offset: usize, bytes: &mut [u8]) -> Result<(), cudaError> {
        assert!(offset + bytes.len() <= self.bytes);

        unsafe {
            check_error(cudaMemcpy(
                bytes.as_mut_ptr() as *mut c_void,
                self.pointer.offset(offset as isize),
                bytes.len(),
                cudaMemcpyKind::cudaMemcpyDeviceToHost,
            ))?;
        }

        Ok(())
    }

    /// writes bytes length of bytes to buffer using an offset
    /// panics if it offset + bytes length overflows the buffer size
    pub fn set(&mut self, offset: usize, bytes: &[u8]) -> Result<(), cudaError> {
        assert!(offset + bytes.len() <= self.bytes);

        unsafe {
            check_error(cudaMemcpy(
                self.pointer.offset(offset as isize),
                bytes.as_ptr() as *const c_void,
                bytes.len(),
                cudaMemcpyKind::cudaMemcpyHostToDevice,
            ))?;
        }

        Ok(())
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe {
            check_error(cudaFree(self.pointer)).unwrap();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::slice;

    #[test]
    fn write_read() {
        let mut buffer = Buffer::new(1).unwrap();
        buffer.set(0, &[0b0110_1001]).unwrap();
        let mut b: u8 = 0;
        buffer.get(0, slice::from_mut(&mut b)).unwrap();
        assert_eq!(b, 0b0110_1001);
    }
}
