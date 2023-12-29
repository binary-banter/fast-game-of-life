use cuda_interface::args::Args;
use cuda_interface::buffer::Buffer;
use cuda_interface::kernel::Kernel;
use cuda_interface::stream::Stream;
use once_cell::sync::Lazy;
use std::fmt::{Display, Formatter};
use std::iter::{once, repeat};
use std::mem;
use std::ops::Deref;
use std::os::raw::c_void;
use toml::Table;

static SETTINGS: Lazy<Table> =
    Lazy::new(|| include_str!("../settings.toml").parse::<Table>().unwrap());
static WORK_GROUP_SIZE: Lazy<usize> =
    Lazy::new(|| SETTINGS["cuda"]["work-group-size"].as_integer().unwrap() as usize);
static WORK_PER_THREAD: Lazy<usize> =
    Lazy::new(|| SETTINGS["cuda"]["work-per-thread"].as_integer().unwrap() as usize);
static STEP_SIZE: Lazy<usize> =
    Lazy::new(|| SETTINGS["cuda"]["step-size"].as_integer().unwrap() as usize);

/// The `Game` struct represents the state of an instance of Conway's Game of Life.
pub struct Game {
    /// Height of the simulated field including padding.
    padded_height: usize,
    /// Width of the simulated field in columns, including padding.
    /// A column is a `u32`, which represents 32 horizontally adjacent cells.
    padded_columns: usize,
    /// The number of work groups in each dimension.
    grid_dim: (u32, u32, u32),
    /// The number of threads per work group in each dimension.
    block_dim: (u32, u32, u32),
    /// Structure containing a pointer to the compiled kernel.
    kernel: Kernel,
    /// Structure containing a pointer to a CUDA stream.
    stream: Stream,
    /// Buffer containing the field, including padding.
    buffer: Buffer<u32>,
    /// Auxiliary buffer used for pointer swapping.
    buffer_aux: Buffer<u32>,
}

// External function declaration for the CUDA kernel.
#[link(name = "kernel", kind = "static")]
extern "C" {
    fn step() -> c_void;
}

impl Game {
    pub fn step(&mut self, steps: u32) {
        let height = self.padded_height as u32;
        let step_size = *STEP_SIZE as u32;

        let chunks = repeat(step_size)
            .take((steps / step_size) as usize)
            .chain(once(steps % step_size).filter(|n| *n != 0));

        for steps in chunks {
            let mut args = Args::default();
            args.add_arg(&mut self.buffer);
            args.add_arg(&mut self.buffer_aux);
            args.add_arg(&height);
            args.add_arg(&steps);
            self.stream
                .launch(&self.kernel, self.grid_dim, self.block_dim, &args, 0)
                .unwrap();
            mem::swap(&mut self.buffer, &mut self.buffer_aux);
        }

        self.stream.wait().unwrap();
    }

    pub fn new(width: usize, height: usize) -> Self {
        let step_size = *STEP_SIZE;
        let work_group_size = *WORK_GROUP_SIZE;
        let work_per_thread = *WORK_PER_THREAD;

        assert!(
            (1..=16).contains(&step_size),
            "Must simulate between 1 and 16 steps at a time."
        );

        // Calculate the number of required work groups horizontally. Each column is 32 bits.
        let horizontal_groups = width.div_ceil(32);

        // Calculate the number of "cleanly" simulated rows per work group.
        let simulated_rows_per_group = work_group_size * work_per_thread - 2 * step_size;

        // Calculate number of required work groups vertically. We need to simulate the entire height "cleanly".
        let vertical_groups = height.div_ceil(simulated_rows_per_group);

        // Calculate number of columns and height with padding.
        let padded_columns = horizontal_groups + 2;
        let padded_height = vertical_groups * simulated_rows_per_group + 2 * step_size;

        let grid_dim = (horizontal_groups as u32, vertical_groups as u32, 1);
        let block_dim = (1, work_group_size as u32, 1);

        let kernel = Kernel::new(step);
        let stream = Stream::new().unwrap();

        // Allocate and zero buffer and auxiliary buffer with enough space for the field plus padding.
        let mut buffer = Buffer::new(padded_columns * padded_height).unwrap();
        buffer.zero().unwrap();
        let mut buffer_aux = Buffer::new(padded_columns * padded_height).unwrap();
        buffer_aux.zero().unwrap();

        Game {
            grid_dim,
            block_dim,
            padded_height,
            padded_columns,
            kernel,
            stream,
            buffer,
            buffer_aux,
        }
    }

    pub fn set(&mut self, x: usize, y: usize) {
        let column = x / 32 + 1;
        let nibble = 0x8000_0000 >> (x % 32);
        let i = (y + STEP_SIZE.deref()) + column * self.padded_height;
        let mut word = 0;
        self.buffer
            .read(i, std::slice::from_mut(&mut word))
            .unwrap();
        word |= nibble;
        self.buffer.write(i, std::slice::from_ref(&word)).unwrap();
    }

    pub fn get(&self, x: usize, y: usize) -> bool {
        let column = x / 32 + 1;
        let nibble = 0x8000_0000 >> (x % 32);
        let i = (y + STEP_SIZE.deref()) + column * self.padded_height;
        let mut word = 0;
        self.buffer
            .read(i, std::slice::from_mut(&mut word))
            .unwrap();
        word & nibble != 0
    }
}

impl Display for Game {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut frame = String::new();

        for y in 0..self.padded_height - 2 * STEP_SIZE.deref() {
            for x in 0..(self.padded_columns - 2) * 32 {
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
