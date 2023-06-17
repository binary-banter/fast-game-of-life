use cuda_interface::args::Args;
use cuda_interface::buffer::Buffer;
use cuda_interface::kernel::Kernel;
use cuda_interface::stream::Stream;
use lazy_static::lazy_static;
use std::fmt::{Display, Formatter};
use std::mem;
use std::mem::size_of;
use std::os::raw::c_void;
use toml::Table;

lazy_static! {
    static ref SETTINGS: Table = include_str!("../settings.toml").parse::<Table>().unwrap();
    static ref WORK_GROUP_SIZE: usize =
        SETTINGS["cuda"]["work-group-size"].as_integer().unwrap() as usize;
    static ref WORK_PER_THREAD: usize =
        SETTINGS["cuda"]["work-per-thread"].as_integer().unwrap() as usize;
    static ref PADDING_X: usize = SETTINGS["cuda"]["padding-x"].as_integer().unwrap() as usize;
    static ref PADDING_Y: usize = SETTINGS["cuda"]["padding-y"].as_integer().unwrap() as usize;
}

/// The `Game` struct stores the state of an instance of Conway's Game of Life.
/// The underlying datatype for this struct is a u32.
pub struct Game {
    /// This field represents the height of the `field` including padding.
    height: usize,

    /// This field represents the width of the `field` in u64s including padding.
    columns: usize,

    grid_dim: (u32, u32, u32),
    block_dim: (u32, u32, u32),

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

impl Game {
    pub fn step(&mut self, steps: u32) {
        let height = self.height as u32;
        let step_size = 16;
        for _ in 0..steps / 16 {
            let mut args = Args::default();
            args.add_arg(&mut self.field_buffer);
            args.add_arg(&mut self.new_field_buffer);
            args.add_arg(&height);
            args.add_arg(&step_size);
            self.stream
                .launch(&self.kernel, self.grid_dim, self.block_dim, &args, 0)
                .unwrap();
            mem::swap(&mut self.field_buffer, &mut self.new_field_buffer);
        }

        let remaining_steps = steps % 16;
        if remaining_steps == 0 {
            self.stream.wait().unwrap();
            return;
        }

        let mut args = Args::default();
        args.add_arg(&mut self.field_buffer);
        args.add_arg(&mut self.new_field_buffer);
        args.add_arg(&height);
        args.add_arg(&remaining_steps);
        self.stream
            .launch(&self.kernel, self.grid_dim, self.block_dim, &args, 0)
            .unwrap();
        mem::swap(&mut self.field_buffer, &mut self.new_field_buffer);

        self.stream.wait().unwrap();
    }

    pub fn new(width: usize, height: usize) -> Self {
        let columns = div_ceil(width, 32) + 2 * *PADDING_X;
        let height = div_ceil(
            div_ceil(height, *WORK_GROUP_SIZE * *WORK_PER_THREAD - 2 * *PADDING_Y)
                * (*WORK_GROUP_SIZE * *WORK_PER_THREAD - 2 * *PADDING_Y),
            *WORK_PER_THREAD,
        ) * *WORK_PER_THREAD
            + 2 * *PADDING_Y;

        let global_work_size_x = columns - 2 * *PADDING_X;
        let global_work_size_y =
            (height - 2 * *PADDING_Y) / (*WORK_GROUP_SIZE * *WORK_PER_THREAD - 2 * *PADDING_Y);
        let local_work_size_y = *WORK_GROUP_SIZE;

        let grid_dim = (global_work_size_x as u32, global_work_size_y as u32, 1);
        let block_dim = (1, local_work_size_y as u32, 1);

        let kernel = Kernel::new(step);
        let stream = Stream::new().unwrap();

        let mut field_buffer = Buffer::new(columns * height).unwrap();
        field_buffer.zero().unwrap();
        let mut new_field_buffer = Buffer::new(columns * height).unwrap();
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
        let column = x / 32 + *PADDING_X;
        let nibble = 0x8000_0000 >> (x % 32);
        let i = (y + *PADDING_Y) + column * self.height;
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
        let column = x / 32 + *PADDING_X;
        let nibble = 0x8000_0000 >> (x % 32);
        let i = (y + *PADDING_Y) + column * self.height;
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

        for y in 0..self.height - 2 * *PADDING_Y {
            for x in 0..(self.columns - 2 * *PADDING_X) * 32 {
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
