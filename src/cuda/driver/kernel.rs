use std::os::raw::c_void;

#[derive(Copy, Clone)]
pub struct Kernel(*const c_void);

impl Kernel {
    pub fn new(function: unsafe extern "C" fn() -> c_void) -> Self {
        Self(function as *const c_void)
    }

    pub fn function(&self) -> *const c_void {
        self.0
    }
}
