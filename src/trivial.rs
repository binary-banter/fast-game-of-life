use std::fmt::{Display, Formatter};
use std::mem::swap;

pub struct Game {
    field: Vec<bool>,
    new_field: Vec<bool>,
    height: usize,
    columns: usize,
}

impl Game {
    pub fn new(width: usize, height: usize) -> Self {
        let columns = width + 2;
        let height = height + 2;
        Game {
            field: vec![false; columns * height],
            new_field: vec![false; columns * height],
            height,
            columns,
        }
    }

    pub fn set(&mut self, x: usize, y: usize) {
        self.field[(y + 1) * self.columns + (x + 1)] = true;
    }

    pub fn get(&self, x: usize, y: usize) -> bool {
        self.field[(y + 1) * self.columns + (x + 1)]
    }

    pub fn step(&mut self, steps: u32) {
        for _ in 0..steps {
            for y in 1..self.height - 1 {
                for x in 1..self.columns - 1 {
                    let i = y * self.columns + x;

                    let nbs = [
                        self.field[i - self.columns - 1], // top left
                        self.field[i - self.columns],     // top
                        self.field[i - self.columns + 1], // top right
                        self.field[i - 1],                // middle left
                        self.field[i + 1],                // middle right
                        self.field[i + self.columns - 1], // bottom left
                        self.field[i + self.columns],     // bottom
                        self.field[i + self.columns + 1], // bottom right
                    ]
                    .into_iter()
                    .filter(|&b| b)
                    .count();

                    self.new_field[i] = nbs == 3 || (self.field[i] && nbs == 2);
                }
            }

            swap(&mut self.field, &mut self.new_field);
        }
    }
}

impl Display for Game {
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
