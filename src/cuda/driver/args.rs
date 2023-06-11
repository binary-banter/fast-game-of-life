use std::marker::PhantomData;
use std::os::raw::c_void;

pub struct Args<'a> {
    args: Vec<*mut c_void>,
    phantom: PhantomData<&'a ()>,
}

//[&mut buffer.pointer as *mut _ as *mut c_void].as_mut_ptr()
impl<'a> Args<'a> {
    pub fn new() -> Self {
        Args {
            args: Default::default(),
            phantom: Default::default(),
        }
    }

    pub fn add_arg(&mut self, arg: impl ToArg) {
        self.args.push(arg.to_arg());
    }

    pub fn as_args(&self) -> *mut *mut c_void {
        self.args.as_ptr() as *mut *mut c_void
    }
}

pub trait ToArg {
    fn to_arg(self) -> *mut c_void;
}

impl ToArg for &mut u32 {
    fn to_arg(self) -> *mut c_void {
        self as *mut u32 as *mut c_void
    }
}