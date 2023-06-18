use rayon::{ThreadPool, ThreadPoolBuilder};
use std::fmt::{Display, Formatter};
use std::mem::swap;
use std::simd::{LaneCount, Simd, SupportedLaneCount};
use std::thread::available_parallelism;

pub struct Game<const N: usize = 8>
where
    LaneCount<N>: SupportedLaneCount,
{
    pool: ThreadPool,
    field: Vec<u64>,
    new_field: Vec<u64>,
    height: usize,
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
        let threads = available_parallelism().unwrap().into();
        let pool = ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        let columns = div_ceil(div_ceil(width, 64), N) * N + 2;
        let height = height + 2;
        Game {
            pool,
            field: vec![0; columns * height],
            new_field: vec![0; columns * height],
            height,
            columns,
        }
    }

    pub fn set(&mut self, x: usize, y: usize) {
        let column = x / 64 + 1;
        let bit = 0x8000_0000_0000_0000 >> (x % 64);
        self.field[(y + 1) * self.columns + column] |= bit;
    }

    pub fn get(&self, x: usize, y: usize) -> bool {
        let column = x / 64 + 1;
        let bit = 0x8000_0000_0000_0000 >> (x % 64);
        (self.field[(y + 1) * self.columns + column] & bit) != 0
    }

    fn sub_step(mut center: Simd<u64, N>, nbs: &[Simd<u64, N>; 8]) -> Simd<u64, N> {
        // stage 0
        let ta0 = nbs[0] ^ nbs[1];
        let a8 = ta0 ^ nbs[2];
        let b0 = (nbs[0] & nbs[1]) | (ta0 & nbs[2]);

        let ta3 = nbs[3] ^ nbs[4];
        let a9 = ta3 ^ nbs[5];
        let b1 = (nbs[3] & nbs[4]) | (ta3 & nbs[5]);

        let aa = nbs[6] ^ nbs[7];
        let b2 = nbs[6] & nbs[7];

        // stage 1
        let ta8 = a8 ^ a9;
        let ab = ta8 ^ aa;
        let b3 = (a8 & a9) | (ta8 & aa);

        let tb0 = b0 ^ b1;
        let b4 = tb0 ^ b2;
        let c0 = (b0 & b1) | (tb0 & b2);

        center |= ab;
        center &= b3 ^ b4;
        center &= !c0;

        return center;
    }

    fn get_simd(field: &[u64], i: usize) -> Simd<u64, N> {
        Simd::from_slice(&field[i..i + N])
    }

    pub fn step(&mut self, steps: u32) {
        for _ in 0..steps {
            let threads = self.pool.current_num_threads();
            let simulation_rows = self.height - 2;
            let chunk_size = (simulation_rows + threads - 1) / threads;

            self.pool.scope(|scope| {
                for (i, target) in self.new_field
                    [self.columns..self.columns * self.height - self.columns]
                    .chunks_mut(chunk_size * self.columns)
                    .enumerate()
                {
                    let columns = &self.columns;
                    let field = &self.field;
                    scope.spawn(move |_| {
                        for yl in 0..(target.len() / columns) {
                            let y = yl + i * chunk_size + 1;
                            for x in (1..columns - 1).step_by(N) {
                                let i = y * columns + x;

                                let center = Self::get_simd(field, i);

                                let mut nbs = [
                                    shr(Self::get_simd(field, i - columns)), // top left
                                    Self::get_simd(field, i - columns),      // top
                                    shl(Self::get_simd(field, i - columns)), // top right
                                    shr(Self::get_simd(field, i)),           // middle left
                                    shl(Self::get_simd(field, i)),           // middle right
                                    shr(Self::get_simd(field, i + columns)), // bottom left
                                    Self::get_simd(field, i + columns),      // bottom
                                    shl(Self::get_simd(field, i + columns)), // bottom right
                                ];

                                // fix bits in neighbouring columns
                                nbs[0][0] |= (field[i - columns - 1] & 1) << 63; // top left
                                nbs[2][N - 1] |= (field[i - columns + N] & (1 << 63)) >> 63; // top right
                                nbs[3][0] |= (field[i - 1] & 0x1) << 63; // left
                                nbs[4][N - 1] |= (field[i + N] & (1 << 63)) >> 63; // right
                                nbs[5][0] |= (field[i + columns - 1] & 1) << 63; // bottom left
                                nbs[7][N - 1] |= (field[i + columns + N] & (1 << 63)) >> 63; // bottom right

                                target[yl * columns + x..yl * columns + N + x]
                                    .copy_from_slice(Self::sub_step(center, &nbs).as_array());
                            }
                        }
                    });
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

        for y in 0..self.height - 2 {
            for x in 0..self.columns - 2 {
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

pub fn shl<const N: usize>(v: Simd<u64, N>) -> Simd<u64, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    let mut mask = [0x00000_0000_0000_0001; N];
    mask[N - 1] = 0;

    let neighbouring_bits =
        (v >> Simd::splat(63)).rotate_lanes_left::<1>() & Simd::from_array(mask);
    (v << Simd::splat(1)) | neighbouring_bits
}

pub fn shr<const N: usize>(v: Simd<u64, N>) -> Simd<u64, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    let mut mask = [0x8000_0000_0000_0000; N];
    mask[0] = 0;

    let neighbouring_bits =
        (v << Simd::splat(63)).rotate_lanes_right::<1>() & Simd::from_array(mask);
    (v >> Simd::splat(1)) | neighbouring_bits
}
