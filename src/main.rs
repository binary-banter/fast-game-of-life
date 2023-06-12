use gol_cuda::game::Game;

pub fn main() {
    let mut game = Game::new(16, 16);
    game.set(4, 5);
    game.set(5, 6);
    game.set(6, 4);
    game.set(6, 5);
    game.set(6, 6);
    game.step(1);
    println!("{}", game);
}
