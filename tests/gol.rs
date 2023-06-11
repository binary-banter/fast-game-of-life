use gol_cuda::game::Game;

#[test]
fn off_0_neighbors() {
    let mut test = Game::new(16, 16);
    test.step(1);
}

#[test]
fn on_0_neighbors_on() {
    let mut test = Game::new(16, 16);
    test.step(1);
}

#[test]
fn off_1_neighbors() {
    let mut test = Game::new(16, 16);
    test.step(1);
}

#[test]
fn on_1_neighbors_on() {
    let mut test = Game::new(16, 16);
    test.step(1);
}

#[test]
fn off_2_neighbors() {
    let mut test = Game::new(16, 16);
    test.step(1);
}

#[test]
fn on_2_neighbors_on() {
    let mut test = Game::new(16, 16);
    test.step(1);
}

#[test]
fn off_3_neighbors() {
    let mut test = Game::new(16, 16);
    test.step(1);
}

#[test]
fn on_3_neighbors_on() {
    let mut test = Game::new(16, 16);
    test.step(1);
}

#[test]
fn off_4_neighbors() {
    let mut test = Game::new(16, 16);
    test.step(1);
}

#[test]
fn on_4_neighbors_on() {
    let mut test = Game::new(16, 16);
    test.step(1);
}

#[test]
fn off_5_neighbors() {
    let mut test = Game::new(16, 16);
    test.step(1);
}

#[test]
fn on_5_neighbors_on() {
    let mut test = Game::new(16, 16);
    test.step(1);
}

#[test]
fn off_6_neighbors() {
    let mut test = Game::new(16, 16);
    test.step(1);
}

#[test]
fn on_6_neighbors_on() {
    let mut test = Game::new(16, 16);
    test.step(1);
}

#[test]
fn off_7_neighbors() {
    let mut test = Game::new(16, 16);
    test.step(1);
}

#[test]
fn on_7_neighbors_on() {
    let mut test = Game::new(16, 16);
    test.step(1);
}

#[test]
fn off_8_neighbors() {
    let mut test = Game::new(16, 16);
    test.step(1);
}

#[test]
fn on_8_neighbors_on() {
    let mut test = Game::new(16, 16);
    test.step(1);
}

#[test]
fn left_edge_off() {
    let mut test = Game::new(128, 128);
    test.step(1);
}

#[test]
fn right_edge_off() {
    let mut test = Game::new(128, 128);
    test.step(1);
}

#[test]
fn left_edge_on() {
    let mut test = Game::new(128, 128);
    test.step(1);
}

#[test]
fn right_edge_on() {
    let mut test = Game::new(128, 128);
    test.step(1);
}

#[test]
fn test_large_grid() {
    let mut game = Game::new(1024, 1024);
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
    let mut test = Game::new(128, 128);
    test.step(1);
}

#[test]
fn blinker_simple() {
    let mut test = Game::new(128, 128);
    test.step(1);
}

#[test]
fn blinker_boundary() {
    let mut test = Game::new(128, 128);
    test.step(1);
}
