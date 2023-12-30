#[cfg(feature = "cuda")]
use fast_game_of_life::cuda::Game;
#[cfg(feature = "opencl")]
use fast_game_of_life::opencl::Game;
#[cfg(feature = "simd")]
use fast_game_of_life::simd::Game;
use fast_game_of_life::trivial;
#[cfg(not(any(feature = "simd", feature = "cuda", feature = "opencl")))]
use fast_game_of_life::trivial::Game;

#[cfg(feature = "cuda")]
#[test]
fn cuda_equivalent_trivial() {
    let grid = include_str!("grid");

    let mut cuda_game = Game::new(64, 1440);
    let mut trivial_game = trivial::Game::new(64, 1440);

    for (y, row) in grid.lines().enumerate() {
        for (x, cell) in row.chars().enumerate() {
            if let '1' = cell {
                cuda_game.set(x, y);
                trivial_game.set(x, y);
            }
        }
    }

    cuda_game.step(50);
    trivial_game.step(50);

    assert_eq!(format!("{cuda_game}"), format!("{trivial_game}"));
}

#[test]
fn off_0_neighbors() {
    let mut game: Game = Game::new(16, 16);
    game.step(1);
    assert!(!game.get(1, 1))
}

#[test]
fn on_0_neighbors_on() {
    let mut game: Game = Game::new(16, 16);
    game.set(1, 1);
    game.step(1);
    assert!(!game.get(1, 1))
}

#[test]
fn off_1_neighbors() {
    let mut game: Game = Game::new(16, 16);
    game.set(0, 0);
    game.step(1);
    assert!(!game.get(1, 1))
}

#[test]
fn on_1_neighbors_on() {
    let mut game: Game = Game::new(16, 16);
    game.set(0, 0);
    game.set(1, 1);
    game.step(1);
    assert!(!game.get(1, 1))
}

#[test]
fn off_2_neighbors() {
    let mut game: Game = Game::new(16, 16);
    game.set(0, 0);
    game.set(1, 0);
    game.step(1);
    assert!(!game.get(1, 1))
}

#[test]
fn on_2_neighbors_on() {
    let mut game: Game = Game::new(16, 16);
    game.set(0, 0);
    game.set(1, 0);
    game.set(1, 1);
    game.step(1);
    assert!(game.get(1, 1))
}

#[test]
fn off_3_neighbors() {
    let mut game: Game = Game::new(16, 16);
    game.set(0, 0);
    game.set(1, 0);
    game.set(2, 0);
    game.step(1);
    assert!(game.get(1, 1))
}

#[test]
fn on_3_neighbors_on() {
    let mut game: Game = Game::new(16, 16);
    game.set(0, 0);
    game.set(1, 0);
    game.set(2, 0);
    game.set(1, 1);
    game.step(1);
    assert!(game.get(1, 1))
}

#[test]
fn off_4_neighbors() {
    let mut game: Game = Game::new(16, 16);
    game.set(0, 0);
    game.set(1, 0);
    game.set(2, 0);
    game.set(0, 1);
    game.step(1);
    assert!(!game.get(1, 1))
}

#[test]
fn on_4_neighbors_on() {
    let mut game: Game = Game::new(16, 16);
    game.set(0, 0);
    game.set(1, 0);
    game.set(2, 0);
    game.set(0, 1);
    game.set(1, 1);
    game.step(1);
    assert!(!game.get(1, 1))
}

#[test]
fn off_5_neighbors() {
    let mut game: Game = Game::new(16, 16);
    game.set(0, 0);
    game.set(1, 0);
    game.set(2, 0);
    game.set(0, 1);
    game.set(2, 1);
    game.step(1);
    assert!(!game.get(1, 1))
}

#[test]
fn on_5_neighbors_on() {
    let mut game: Game = Game::new(16, 16);
    game.set(0, 0);
    game.set(1, 0);
    game.set(2, 0);
    game.set(0, 1);
    game.set(2, 1);
    game.set(1, 1);
    game.step(1);
    assert!(!game.get(1, 1))
}

#[test]
fn off_6_neighbors() {
    let mut game: Game = Game::new(16, 16);
    game.set(0, 0);
    game.set(1, 0);
    game.set(2, 0);
    game.set(0, 1);
    game.set(2, 1);
    game.set(0, 2);
    game.step(1);
    assert!(!game.get(1, 1))
}

#[test]
fn on_6_neighbors_on() {
    let mut game: Game = Game::new(16, 16);
    game.set(0, 0);
    game.set(1, 0);
    game.set(2, 0);
    game.set(0, 1);
    game.set(2, 1);
    game.set(1, 1);
    game.set(0, 2);
    game.step(1);
    assert!(!game.get(1, 1))
}

#[test]
fn off_7_neighbors() {
    let mut game: Game = Game::new(16, 16);
    game.set(0, 0);
    game.set(1, 0);
    game.set(2, 0);
    game.set(0, 1);
    game.set(2, 1);
    game.set(0, 2);
    game.set(1, 2);
    game.step(1);
    assert!(!game.get(1, 1))
}

#[test]
fn on_7_neighbors_on() {
    let mut game: Game = Game::new(16, 16);
    game.set(0, 0);
    game.set(1, 0);
    game.set(2, 0);
    game.set(0, 1);
    game.set(2, 1);
    game.set(1, 1);
    game.set(0, 2);
    game.set(1, 2);
    game.step(1);
    assert!(!game.get(1, 1))
}

#[test]
fn off_8_neighbors() {
    let mut game: Game = Game::new(16, 16);
    game.set(0, 0);
    game.set(1, 0);
    game.set(2, 0);
    game.set(0, 1);
    game.set(2, 1);
    game.set(0, 2);
    game.set(1, 2);
    game.set(2, 2);
    game.step(1);
    assert!(!game.get(1, 1))
}

#[test]
fn on_8_neighbors_on() {
    let mut game: Game = Game::new(16, 16);
    game.set(0, 0);
    game.set(1, 0);
    game.set(2, 0);
    game.set(0, 1);
    game.set(2, 1);
    game.set(1, 1);
    game.set(0, 2);
    game.set(1, 2);
    game.set(2, 2);
    game.step(1);
    assert!(!game.get(1, 1))
}

#[test]
fn left_edge_off() {
    let mut game: Game = Game::new(128, 128);
    game.set(63, 0);
    game.set(63, 1);
    game.set(63, 2);
    game.step(1);
    assert!(game.get(64, 1))
}

#[test]
fn right_edge_off() {
    let mut game: Game = Game::new(128, 128);
    game.set(64, 0);
    game.set(64, 1);
    game.set(64, 2);
    game.step(1);
    assert!(game.get(63, 1))
}

#[test]
fn left_edge_on() {
    let mut game: Game = Game::new(128, 128);
    game.set(63, 0);
    game.set(63, 1);
    game.set(63, 2);
    game.set(64, 1);
    game.set(65, 0);
    game.set(65, 1);
    game.set(65, 2);
    game.step(1);
    assert!(!game.get(64, 1))
}

#[test]
fn right_edge_on() {
    let mut game: Game = Game::new(128, 128);
    game.set(64, 0);
    game.set(64, 1);
    game.set(64, 2);
    game.set(63, 1);
    game.set(62, 0);
    game.set(62, 1);
    game.set(62, 2);
    game.step(1);
    assert!(!game.get(63, 1))
}

#[test]
fn test_large_grid() {
    let mut game: Game = Game::new(1024, 1024);
    const STEPS: usize = 1000;
    game.set(4, 5);
    game.set(5, 6);
    game.set(6, 4);
    game.set(6, 5);
    game.set(6, 6);
    game.step(4 * STEPS as u32);
    assert!(game.get(4 + STEPS, 5 + STEPS));
    assert!(game.get(5 + STEPS, 6 + STEPS));
    assert!(game.get(6 + STEPS, 4 + STEPS));
    assert!(game.get(6 + STEPS, 5 + STEPS));
    assert!(game.get(6 + STEPS, 6 + STEPS));
}

#[test]
fn test_large_grid_blinker() {
    let mut game: Game = Game::new(128, 128);
    game.set(62, 2);
    game.set(63, 2);
    game.set(64, 2);
    game.step(2);
    assert!(game.get(62, 2));
    assert!(game.get(63, 2));
    assert!(game.get(64, 2));
}

#[test]
fn blinker_simple() {
    let mut game: Game = Game::new(128, 128);
    game.set(32, 0);
    game.set(32, 1);
    game.set(32, 2);
    game.step(64);
    assert!(game.get(32, 0));
    assert!(game.get(32, 1));
    assert!(game.get(32, 2));
}

#[test]
fn blinker_boundary() {
    let mut game: Game = Game::new(128, 128);
    game.set(64, 63);
    game.set(64, 64);
    game.set(64, 65);
    game.step(64);
    assert!(game.get(64, 63));
    assert!(game.get(64, 64));
    assert!(game.get(64, 65));
}
