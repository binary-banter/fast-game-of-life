pub mod driver;

use crate::cuda::driver::buffer::Buffer;
use crate::cuda::driver::kernel::Kernel;
use crate::cuda::driver::stream::Stream;
use crate::game::driver::args::Args;
use cuda_runtime_sys::dim3;
use std::ffi::c_int;
use std::fmt::{Display, Formatter};
use std::mem::size_of;
use std::os::raw::{c_uint, c_void};
use std::{mem, ptr};

/// The `Game` struct stores the state of an instance of Conway's Game of Life.
///
/// The underlying datatype for this struct is a u64.
pub struct Game {
    /// This field represents the height of the `field` including padding.
    height: usize,

    /// This field represents the width of the `field` in u64s including padding.
    columns: usize,

    grid_dim: dim3,
    block_dim: dim3,

    kernel: Kernel,
    stream: Stream,
    field_buffer: Buffer<u32>,
    new_field_buffer: Buffer<u32>,
}

#[link(name = "kernel", kind = "static")]
extern "C" {
    fn step() -> c_void;
}

fn div_ceil(x: usize, y: usize) -> usize {
    (x + y - 1) / y
}

const WORK_GROUP_SIZE: usize = 512;
const WORK_PER_THREAD: usize = 3;

const PADDING_X: usize = 1;
const PADDING_Y: usize = 16;

impl Game {
    pub fn step(&mut self, steps: u32) {
        for _ in 0..steps / 16 {
            let mut args = Args::new();
            args.add_arg(&mut self.field_buffer);
            args.add_arg(&mut self.new_field_buffer);
            args.add_arg(&mut (self.height as u32));
            args.add_arg(&mut 16u32);
            self.stream
                .launch(
                    self.kernel,
                    self.grid_dim,
                    self.block_dim,
                    &args,
                    2 * (WORK_GROUP_SIZE * WORK_PER_THREAD + 2) * size_of::<u32>(),
                )
                .unwrap();
            mem::swap(&mut self.field_buffer, &mut self.new_field_buffer);
        }

        let remaining_steps = steps % 16;
        if remaining_steps == 0 {
            return;
        }

        let mut args = Args::new();
        args.add_arg(&mut self.field_buffer);
        args.add_arg(&mut self.new_field_buffer);
        args.add_arg(&mut (self.height as u32));
        args.add_arg(&mut (remaining_steps as u32));
        self.stream
            .launch(
                self.kernel,
                self.grid_dim,
                self.block_dim,
                &args,
                2 * (WORK_GROUP_SIZE * WORK_PER_THREAD + 2) * size_of::<u32>(),
            )
            .unwrap();
        mem::swap(&mut self.field_buffer, &mut self.new_field_buffer);

        self.stream.wait().unwrap();
    }

    pub fn new(width: usize, height: usize) -> Self {
        let columns = div_ceil(width, 32) + 2 * PADDING_X;
        let height = div_ceil(
            div_ceil(height, WORK_GROUP_SIZE * WORK_PER_THREAD - 2 * PADDING_Y)
                * (WORK_GROUP_SIZE * WORK_PER_THREAD - 2 * PADDING_Y),
            WORK_PER_THREAD,
        ) * WORK_PER_THREAD
            + 2 * PADDING_Y;

        let global_work_size_x = columns - 2 * PADDING_X;
        let global_work_size_y =
            (height - 2 * PADDING_Y) / (WORK_GROUP_SIZE * WORK_PER_THREAD - 2 * PADDING_Y);
        let local_work_size_y = WORK_GROUP_SIZE;

        let grid_dim = dim3 {
            x: global_work_size_x as c_uint,
            y: global_work_size_y as c_uint,
            z: 1,
        };
        let block_dim = dim3 {
            x: 1,
            y: local_work_size_y as c_uint,
            z: 1,
        };

        let kernel = Kernel::new(step);
        let stream = Stream::new().unwrap();

        let mut field_buffer = Buffer::new(columns * height * size_of::<u32>()).unwrap();
        field_buffer.zero().unwrap();
        let mut new_field_buffer = Buffer::new(columns * height * size_of::<u32>()).unwrap();
        new_field_buffer.zero().unwrap();

        Game {
            grid_dim,
            block_dim,
            height,
            columns,
            kernel,
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
        self.field_buffer
            .read(i, std::slice::from_mut(&mut word))
            .unwrap();
        word |= nibble;
        self.field_buffer
            .write(i, std::slice::from_ref(&word))
            .unwrap();
    }

    pub fn get(&self, x: usize, y: usize) -> bool {
        let column = x / 32 + PADDING_X;
        let nibble = 0x8000_0000 >> (x % 32);
        let i = (y + PADDING_Y) + column * self.height;
        let mut word: u32 = 0;
        self.field_buffer
            .read(i, std::slice::from_mut(&mut word))
            .unwrap();
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
