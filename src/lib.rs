#[cfg(not(feature = "cuda"))]
pub mod opencl;

#[cfg(not(feature = "cuda"))]
pub use opencl as game;

#[cfg(feature = "cuda")]
pub mod cuda;

#[cfg(feature = "cuda")]
pub use cuda as game;
