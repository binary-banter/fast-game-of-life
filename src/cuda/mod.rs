use rustacuda::context::{Context, ContextFlags};
use rustacuda::device::Device;
use rustacuda::memory::{CopyDestination, DeviceBuffer};
use rustacuda::module::Module;
use rustacuda::stream::{Stream, StreamFlags};
use rustacuda::{launch, CudaFlags};
use std::ffi::CString;
use std::fmt::{Display, Formatter};
use std::mem;
use std::ops::{Index, IndexMut};

/// The `Game` struct stores the state of an instance of Conway's Game of Life.
///
/// The underlying datatype for this struct is a u64.
#[derive(Debug)]
pub struct Game {
    /// This field represents the height of the `field` including padding.
    height: usize,

    /// This field represents the width of the `field` in u64s including padding.
    columns: usize,

    field_buffer: DeviceBuffer<u32>,
    new_field_buffer: DeviceBuffer<u32>,

    stream: Stream,
    module: Module,
    _ctx: Context,
    device: Device,
}

fn div_ceil(x: usize, y: usize) -> usize {
    (x + y - 1) / y
}

const KERNEL_NAME: &str = "step";
const OPTIMIZATIONS: &str = "";
const WORK_GROUP_SIZE: usize = 256;
const WORK_PER_THREAD: usize = 1;

const PADDING_X: usize = 1;
const PADDING_Y: usize = 16;

impl Game {
    pub fn step(&mut self, steps: u32) {
        let module = &self.module;
        let stream = &self.stream;
        let blocks_y = (self.height - 2 * PADDING_Y) as u32;
        let blocks_x = (self.columns - 2 * PADDING_X) as u32;

        for _ in 0..steps {
            //<<grid, block, shared_memory_size, stream>>>

            unsafe {
                launch!(module.step<<<(blocks_y, blocks_x),(1,1),0,stream>>>(
                    self.field_buffer.as_device_ptr(),
                    self.new_field_buffer.as_device_ptr()
                )).unwrap();
            }
            mem::swap(&mut self.field_buffer, &mut self.new_field_buffer);
        }

        stream.synchronize().unwrap();
    }

    pub fn new(width: usize, height: usize) -> Self {
        // Set up the context, load the module, and create a stream to run kernels in.
        rustacuda::init(CudaFlags::empty()).unwrap();
        let device = Device::get_device(0).unwrap();
        let ctx =
            Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)
                .unwrap();

        let ptx = CString::new(include_str!("../../target/gol.ptx")).unwrap();
        let module = Module::load_from_string(&ptx).unwrap();
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None).unwrap();

        let columns = div_ceil(width, 32) + 2 * PADDING_X;
        let height = div_ceil(
            div_ceil(height, WORK_GROUP_SIZE * WORK_PER_THREAD - 2 * PADDING_Y)
                * (WORK_GROUP_SIZE * WORK_PER_THREAD - 2 * PADDING_Y),
            WORK_PER_THREAD,
        ) * WORK_PER_THREAD
            + 2 * PADDING_Y;

        let field_buffer = unsafe {
            DeviceBuffer::zeroed(columns * height)
        }.unwrap();

        let new_field_buffer = unsafe {
            DeviceBuffer::zeroed(columns * height)
        }.unwrap();

        Game {
            height,
            columns,
            _ctx: ctx,
            device,
            module,
            stream,
            field_buffer,
            new_field_buffer,
        }
    }

    pub fn set(&mut self, x: usize, y: usize) {
        let column = x / 32 + PADDING_X;
        let nibble = 0x8000_0000 >> (x % 32);
        let i = (y + PADDING_Y) + column * self.height;
        let mut word = 0;
        self.field_buffer[i..i+1].copy_to(std::slice::from_mut(&mut word)).unwrap();
        word |= nibble;
        self.field_buffer[i..i+1].copy_from(std::slice::from_ref(&word)).unwrap();
    }

    pub fn get(&self, x: usize, y: usize) -> bool {
        let column = x / 32 + PADDING_X;
        let nibble = 0x8000_0000 >> (x % 32);
        let i = (y + PADDING_Y) + column * self.height;
        let mut word= 0;
        self.field_buffer[i..i+1].copy_to(std::slice::from_mut(&mut word)).unwrap();
        word & nibble != 0
    }
}

impl Display for Game {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut frame = String::new();

        for y in 0..self.height - 2 * PADDING_Y {
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
