use opencl3::command_queue::{CommandQueue, CL_QUEUE_PROFILING_ENABLE};
use opencl3::context::Context;
use opencl3::device::{get_all_devices, Device, CL_DEVICE_TYPE_GPU};
use opencl3::kernel::{ExecuteKernel, Kernel};
use opencl3::memory::{Buffer, CL_MEM_READ_WRITE};
use opencl3::program::Program;
use opencl3::types::CL_BLOCKING;
use std::fmt::{Display, Formatter};
use std::mem::size_of;
use std::{mem, ptr};

/// The `Game` struct stores the state of an instance of Conway's Game of Life.
///
/// The underlying datatype for this struct is a u64.
#[derive(Debug)]
pub struct Game {
    /// This field represents the height of the `field` including padding.
    pub height: usize,

    /// This field represents the width of the `field` in u64s including padding.
    columns: usize,

    _context: Context,
    queue: CommandQueue,
    kernel: Kernel,
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
        let global_work_size = (self.height - 2 * PADDING_Y) / (WORK_GROUP_SIZE*WORK_PER_THREAD - 2 * PADDING_Y) * WORK_GROUP_SIZE;
        if let Some(event) = (0..steps/16).map(|_| {
            let event = unsafe {
                ExecuteKernel::new(&self.kernel)
                    .set_arg(&self.field_buffer)
                    .set_arg(&self.new_field_buffer)
                    .set_arg(&(self.height as u32))
                    .set_arg(&16u32)
                    .set_global_work_sizes(&vec![global_work_size, self.columns - 2 * PADDING_X])
                    .set_global_work_offsets(&vec![0, PADDING_X])
                    .set_local_work_sizes(&vec![WORK_GROUP_SIZE,1])
                    .enqueue_nd_range(&self.queue).unwrap()
            };
            mem::swap(&mut self.field_buffer, &mut self.new_field_buffer);
            event
        }).last() {
            event.wait().unwrap();
        }

        let remaining_steps = steps % 16;
        if remaining_steps == 0 {
            return
        }

        unsafe {
            ExecuteKernel::new(&self.kernel)
                .set_arg(&self.field_buffer)
                .set_arg(&self.new_field_buffer)
                .set_arg(&(self.height as u32))
                .set_arg(&remaining_steps)
                .set_global_work_sizes(&vec![global_work_size, self.columns - 2 * PADDING_X])
                .set_global_work_offsets(&vec![0, PADDING_X])
                .set_local_work_sizes(&vec![WORK_GROUP_SIZE, 1])
                .enqueue_nd_range(&self.queue).unwrap()
        }.wait().unwrap();
        mem::swap(&mut self.field_buffer, &mut self.new_field_buffer);
    }

    pub fn new(width: usize, height: usize) -> Self {
        let device_id = *get_all_devices(CL_DEVICE_TYPE_GPU)
            .unwrap()
            .first()
            .expect("no device found in platform");
        let device = Device::new(device_id);

        let context = Context::from_device(&device).expect("Context::from_device failed");

        #[allow(deprecated)]
            let queue = CommandQueue::create_default(&context, CL_QUEUE_PROFILING_ENABLE)
            .expect("CommandQueue::create_default failed");

        let program = Program::create_and_build_from_source(
            &context,
            include_str!("../kernels/gol.cl"),
            OPTIMIZATIONS,
        )
            .expect("Program::create_and_build_from_source failed");
        let kernel = Kernel::create(&program, KERNEL_NAME).expect("Kernel::create failed");

        let columns = div_ceil(width, 32) + 2 * PADDING_X;
        let height = div_ceil(div_ceil(height, WORK_GROUP_SIZE * WORK_PER_THREAD - 2 * PADDING_Y) * (WORK_GROUP_SIZE * WORK_PER_THREAD - 2 * PADDING_Y), WORK_PER_THREAD) * WORK_PER_THREAD + 2 * PADDING_Y;

        let mut field_buffer = unsafe {
            Buffer::create(
                &context,
                CL_MEM_READ_WRITE,
                columns * height,
                ptr::null_mut(),
            )
                .unwrap()
        };

        let mut new_field_buffer = unsafe {
            Buffer::create(
                &context,
                CL_MEM_READ_WRITE,
                columns * height,
                ptr::null_mut(),
            )
                .unwrap()
        };

        unsafe {
            queue
                .enqueue_fill_buffer(
                    &mut field_buffer,
                    &[0],
                    0,
                    columns * height * size_of::<u32>(),
                    &[],
                )
                .unwrap()
                .wait()
                .unwrap()
        };
        unsafe {
            queue
                .enqueue_fill_buffer(
                    &mut new_field_buffer,
                    &[0],
                    0,
                    columns * height * size_of::<u32>(),
                    &[],
                )
                .unwrap()
                .wait()
                .unwrap()
        };

        Game {
            height,
            columns,
            _context: context,
            queue,
            kernel,
            field_buffer,
            new_field_buffer,
        }
    }

    pub fn set(&mut self, x: usize, y: usize) {
        let column = x / 32 + PADDING_X;
        let nibble = 0x8000_0000 >> (x % 32);
        let i = (y + PADDING_Y) + column * self.height;
        let mut word: u32 = 0;
        unsafe { self.queue.enqueue_read_buffer(&self.field_buffer, CL_BLOCKING, i * size_of::<u32>(), std::slice::from_mut(&mut word), &[]).unwrap(); }
        word |= nibble;
        unsafe { self.queue.enqueue_write_buffer(&mut self.field_buffer, CL_BLOCKING, i * size_of::<u32>(), std::slice::from_ref(&word), &[]).unwrap(); }
    }

    pub fn get(&self, x: usize, y: usize) -> bool {
        let column = x / 32 + PADDING_X;
        let nibble = 0x8000_0000 >> (x % 32);
        let i = (y + PADDING_Y) + column * self.height;
        let mut word: u32 = 0;
        unsafe { self.queue.enqueue_read_buffer(&self.field_buffer, CL_BLOCKING, i * size_of::<u32>(), std::slice::from_mut(&mut word), &[]).unwrap(); }
        word & nibble != 0
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
