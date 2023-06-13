//! This library provides a quick way to simulate dense instances of Conway's Game of Life.
//!
//! To create an instance, create a new Game struct with a provided number of SIMD lanes.
//! To enable cells, use `set`. To do one step, use `step`. To read a particular cell, use `get`.
//!
//! For example:
//! ```
//! use game_of_life::Game;
//!
//! let mut game = Game::<8>::new(128, 128);
//!
//! // glider
//! game.set(4, 5);
//! game.set(5, 6);
//! game.set(6, 4);
//! game.set(6, 5);
//! game.set(6, 6);
//!
//! // blinker
//! game.set(15, 0);
//! game.set(15, 1);
//! game.set(15, 2);
//!
//! game.step();
//!
//! assert!(game.get(15,1));
//! ```

use itertools::Itertools;
use std::fmt::{Display, Formatter};
use std::mem::swap;
use std::simd::{LaneCount, Simd, SupportedLaneCount};

/// The `Game` struct stores the state of an instance of Conway's Game of Life.
///
/// The underlying datatype for this struct is a u64.
#[derive(Debug)]
pub struct Game<const N: usize = 8>
where
    LaneCount<N>: SupportedLaneCount,
{
    /// This field stores the cells in a contiguous array of nibbles,
    /// where 0b0000 represents dead and 0b0001 alive.
    field: Vec<u64>,

    /// This field is used to perform pointer-swapping for performance.
    new_field: Vec<u64>,

    /// This field represents the height of the `field` including padding.
    height: usize,

    /// This field represents the width of the `field` in u64s including padding.
    columns: usize,
}

fn div_ceil(x: usize, y: usize) -> usize {
    (x + y - 1) / y
}

impl<const N: usize> Game<N>
where
    LaneCount<N>: SupportedLaneCount,
{
    pub fn new(width: usize, height: usize) -> Self {
        let columns = div_ceil(div_ceil(width * 4, 64), N) * N + 2;
        let height = height + 2;
        Game {
            field: vec![0; columns * height],
            new_field: vec![0; columns * height],
            height,
            columns,
        }
    }

    pub fn set(&mut self, x: usize, y: usize) {
        let column = x / 16 + 1;
        let nibble = 0x1000_0000_0000_0000 >> ((x % 16) * 4);
        self.field[(y + 1) * self.columns + column] |= nibble;
    }

    pub fn get(&self, x: usize, y: usize) -> bool {
        let column = x / 16 + 1;
        let nibble = 0x1000_0000_0000_0000 >> ((x % 16) * 4);
        self.field[(y + 1) * self.columns + column] & nibble != 0
    }

    pub fn step(&mut self, steps: u32) {
        for _ in 0..steps {
            (self.field.chunks(self.columns))
                .tuple_windows()
                .zip(self.new_field.chunks_mut(self.columns).skip(1))
                .for_each(|((prev, cur, next), new)| {
                    for i in (1..self.columns - 1).step_by(N) {
                        // Logic for common cases
                        let mut count =
                            Simd::from_slice(&prev[i..i + N]) + Simd::from_slice(&next[i..i + N]);
                        let partial = count + Simd::from_slice(&cur[i..i + N]);
                        count += shl_4bit(partial);
                        count += shr_4bit(partial);

                        // Logic for edge cases
                        // add cells on the right in the next block
                        let nibble1 = (prev[i + N] + cur[i + N] + next[i + N]) >> 60;
                        // add cells on the left in the previous block
                        let nibble2 = (prev[i - 1] + cur[i - 1] + next[i - 1]) << 60;
                        let mut arr = [0; N];
                        if N == 1 {
                            arr[0] = nibble1 | nibble2;
                        } else {
                            arr[0] = nibble2;
                            arr[N - 1] = nibble1;
                        }
                        count += Simd::<u64, N>::from_array(arr);

                        let mut result = Simd::from_slice(&cur[i..i + N]);

                        result |= count;
                        result &= count >> Simd::<u64, N>::splat(1);
                        result &= !(count >> Simd::<u64, N>::splat(2));
                        result &= !(count >> Simd::<u64, N>::splat(3));
                        result &= Simd::splat(0x1111_1111_1111_1111);

                        new[i..i + N].copy_from_slice(result.as_array());
                    }
                });

            swap(&mut self.field, &mut self.new_field);
        }
    }
}

impl<const N: usize> Display for Game<N>
where
    LaneCount<N>: SupportedLaneCount,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut frame = String::new();

        for row in 1..self.height - 1 {
            for column in 1..self.columns - 1 {
                for byte in self.field[row * self.columns + column].to_be_bytes() {
                    frame.push_str(match byte {
                        0b0000_0000 => "..",
                        0b0000_0001 => ".█",
                        0b0001_0000 => "█.",
                        0b0001_0001 => "██",
                        _ => unreachable!(),
                    });
                }
            }
            frame.push('\n');
        }

        write!(f, "{frame}")
    }
}

pub fn shl_4bit<const N: usize>(v: Simd<u64, N>) -> Simd<u64, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    let mut mask = [0x00000_0000_0000_000F; N];
    mask[N - 1] = 0;

    let neighbouring_nibbles =
        (v >> Simd::splat(60)).rotate_lanes_left::<1>() & Simd::from_array(mask);
    (v << Simd::splat(4)) | neighbouring_nibbles
}

pub fn shr_4bit<const N: usize>(v: Simd<u64, N>) -> Simd<u64, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    let mut mask = [0xF000_0000_0000_0000; N];
    mask[0] = 0;

    let neighbouring_nibbles =
        (v << Simd::splat(60)).rotate_lanes_right::<1>() & Simd::from_array(mask);
    (v >> Simd::splat(4)) | neighbouring_nibbles
}
