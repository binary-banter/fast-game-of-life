use crate::cuda::driver::args::ToArg;
use crate::cuda::driver::check_error;
use cuda_runtime_sys::{cudaError, cudaFree, cudaMalloc, cudaMemcpy, cudaMemcpyKind, cudaMemset};
use std::os::raw::c_int;
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
    pub fn read(&self, offset: usize, bytes: &mut [u8]) -> Result<(), cudaError> {
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

    pub fn read_all(&self) -> Result<Vec<u8>, cudaError> {
        let mut vec = vec![0; self.bytes];
        self.read(0, &mut vec)?;
        Ok(vec)
    }

    /// writes bytes length of bytes to buffer using an offset
    /// panics if it offset + bytes length overflows the buffer size
    pub fn write(&mut self, offset: usize, bytes: &[u8]) -> Result<(), cudaError> {
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

    pub fn write_multiple(
        &mut self,
        offset: usize,
        count: usize,
        value: u8,
    ) -> Result<(), cudaError> {
        assert!(offset + count <= self.bytes);

        unsafe {
            check_error(cudaMemset(
                self.pointer.offset(offset as isize),
                value as c_int,
                count,
            ))?;
        }

        Ok(())
    }

    pub fn write_all(&mut self, value: u8) -> Result<(), cudaError> {
        self.write_multiple(0, self.bytes, value)
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe {
            check_error(cudaFree(self.pointer)).unwrap();
        }
    }
}

impl ToArg for &mut Buffer {
    fn to_arg(self) -> *mut c_void {
        (&mut self.pointer) as *mut *mut c_void as *mut c_void
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::slice;

    #[test]
    fn write_read() {
        let mut buffer = Buffer::new(1).unwrap();
        buffer.write(0, &[0b0110_1001]).unwrap();
        let mut b: u8 = 0;
        buffer.read(0, slice::from_mut(&mut b)).unwrap();
        assert_eq!(b, 0b0110_1001);
    }
}
