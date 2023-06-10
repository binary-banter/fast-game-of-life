use std::fmt::{Display, Formatter};
use std::mem::size_of;
use std::{mem, ptr};

use rust_gpu_tools::{opencl, program_closures, Device, GPUError, Program};
#[cfg(not(feature = "cuda"))]
use rust_gpu_tools::opencl::{Buffer, Kernel};
#[cfg(feature = "cuda")]
use rust_gpu_tools::cuda;

/// The `Game` struct stores the state of an instance of Conway's Game of Life.
///
/// The underlying datatype for this struct is a u64.
pub struct Game {
    /// This field represents the height of the `field` including padding.
    pub height: usize,

    /// This field represents the width of the `field` in u64s including padding.
    columns: usize,

    device: &'static Device,
    program: Program,
    field_buffer: Buffer<u32>,
    new_field_buffer: Buffer<u32>,
}

fn div_ceil(x: usize, y: usize) -> usize {
    (x + y - 1) / y
}

const KERNEL_NAME: &str = "step";
const OPTIMIZATIONS: &str = "";
const WORK_GROUP_SIZE: usize = 256;
const WORK_PER_THREAD: usize = 3;

const PADDING_X: usize = 1;
const PADDING_Y: usize = 16;

impl Game {
    pub fn step(&mut self, steps: u32) {
        // let global_work_size = (self.height - 2 * PADDING_Y) / (WORK_GROUP_SIZE*WORK_PER_THREAD - 2 * PADDING_Y) * WORK_GROUP_SIZE;
        // if let Some(event) = (0..steps/16).map(|_| {
        //     let event = unsafe {
        //         ExecuteKernel::new(&self.kernel)
        //             .set_arg(&self.field_buffer)
        //             .set_arg(&self.new_field_buffer)
        //             .set_arg(&(self.height as u32))
        //             .set_arg(&16u32)
        //             .set_global_work_sizes(&vec![global_work_size, self.columns - 2 * PADDING_X])
        //             .set_global_work_offsets(&vec![0, PADDING_X])
        //             .set_local_work_sizes(&vec![WORK_GROUP_SIZE,1])
        //             .enqueue_nd_range(&self.queue).unwrap()
        //     };
        //     mem::swap(&mut self.field_buffer, &mut self.new_field_buffer);
        //     event
        // }).last() {
        //     event.wait().unwrap();
        // }
        //
        // let remaining_steps = steps % 16;
        // if remaining_steps == 0 {
        //     return
        // }
        //
        // unsafe {
        //     ExecuteKernel::new(&self.kernel)
        //         .set_arg(&self.field_buffer)
        //         .set_arg(&self.new_field_buffer)
        //         .set_arg(&(self.height as u32))
        //         .set_arg(&remaining_steps)
        //         .set_global_work_sizes(&vec![global_work_size, self.columns - 2 * PADDING_X])
        //         .set_global_work_offsets(&vec![0, PADDING_X])
        //         .set_local_work_sizes(&vec![WORK_GROUP_SIZE, 1])
        //         .enqueue_nd_range(&self.queue).unwrap()
        // }.wait().unwrap();
        // mem::swap(&mut self.field_buffer, &mut self.new_field_buffer);
        todo!()
    }

    pub fn new(width: usize, height: usize) -> Self {
        let device = Device::all()[0];
        let program = get_program(device);

        let columns = div_ceil(width, 32) + 2 * PADDING_X;
        let height = div_ceil(div_ceil(height, WORK_GROUP_SIZE * WORK_PER_THREAD - 2 * PADDING_Y) * (WORK_GROUP_SIZE * WORK_PER_THREAD - 2 * PADDING_Y), WORK_PER_THREAD) * WORK_PER_THREAD + 2 * PADDING_Y;

        let (field_buffer, new_field_buffer) = program.run(program_closures!(|program, _args| -> Result<_, GPUError> {
            //TODO do this using a pattern write
            let vec = vec![0; columns * height];

            let field_buffer: Buffer<u32> = program.create_buffer_from_slice(&vec).unwrap();
            let new_field_buffer: Buffer<u32> = program.create_buffer_from_slice(&vec).unwrap();

            Ok((field_buffer, new_field_buffer))
        }), ()).unwrap();


        // unsafe {
        //     queue
        //         .enqueue_fill_buffer(
        //             &mut field_buffer,
        //             &[0],
        //             0,
        //             columns * height * size_of::<u32>(),
        //             &[],
        //         )
        //         .unwrap()
        //         .wait()
        //         .unwrap()
        // };
        // unsafe {
        //     queue
        //         .enqueue_fill_buffer(
        //             &mut new_field_buffer,
        //             &[0],
        //             0,
        //             columns * height * size_of::<u32>(),
        //             &[],
        //         )
        //         .unwrap()
        //         .wait()
        //         .unwrap()
        // };

        Game {
            height,
            columns,
            device,
            program,
            field_buffer,
            new_field_buffer,
        }
    }

    pub fn set(&mut self, x: usize, y: usize) {
        // let column = x / 32 + PADDING_X;
        // let nibble = 0x8000_0000 >> (x % 32);
        // let i = (y + PADDING_Y) + column * self.height;
        // let mut word: u32 = 0;
        // unsafe { self.queue.enqueue_read_buffer(&self.field_buffer, CL_BLOCKING, i * size_of::<u32>(), std::slice::from_mut(&mut word), &[]).unwrap(); }
        // word |= nibble;
        // unsafe { self.queue.enqueue_write_buffer(&mut self.field_buffer, CL_BLOCKING, i * size_of::<u32>(), std::slice::from_ref(&word), &[]).unwrap(); }
        todo!()
    }

    pub fn get(&self, x: usize, y: usize) -> bool {
        // let column = x / 32 + PADDING_X;
        // let nibble = 0x8000_0000 >> (x % 32);
        // let i = (y + PADDING_Y) + column * self.height;
        // let mut word: u32 = 0;
        // unsafe { self.queue.enqueue_read_buffer(&self.field_buffer, CL_BLOCKING, i * size_of::<u32>(), std::slice::from_mut(&mut word), &[]).unwrap(); }
        // word & nibble != 0
        todo!()
    }
}

impl Display for Game {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut frame = String::new();

        for y in 0..self.height - 2 * PADDING_Y{
            for x in 0..(self.columns - 2 * PADDING_X) * 32 {
                if self.get(x, y) {
                    frame.push('█');
                } else {
                    frame.push('.');
                }
            }
            frame.push('\n');
        }

        write!(f, "{frame}")
    }
}

#[cfg(feature = "cuda")]
fn get_program(device: &Device) -> Program {
    // The kernel was compiled with:
    let cuda_kernel = CString::new("../target/gol.fatbin").unwrap();
    let cuda_device = device.cuda_device().unwrap();
    let cuda_program = cuda::Program::from_binary(cuda_device, &cuda_kernel).unwrap();
    Program::Cuda(cuda_program)
}

#[cfg(not(feature = "cuda"))]
fn get_program(device: &Device) -> Program {
    let opencl_kernel = include_str!("./kernels/gol.cl");
    let opencl_device = device.opencl_device().unwrap();
    let opencl_program = opencl::Program::from_opencl(opencl_device, opencl_kernel).unwrap();
    Program::Opencl(opencl_program)
}