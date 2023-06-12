#[cfg(not(any(feature = "opencl", feature = "cuda", feature = "simd")))]
compile_error!("No features were selected!");

#[cfg(all(feature = "opencl", feature = "cuda"))]
compile_error!("The \"opencl\" and \"cuda\" features are mutually exclusive!");

#[cfg(all(feature = "opencl", feature = "simd"))]
compile_error!("The \"opencl\" and \"simd\" features are mutually exclusive!");

#[cfg(all(feature = "cuda", feature = "simd"))]
compile_error!("The \"cuda\" and \"simd\" features are mutually exclusive!");

#[cfg(feature = "opencl")]
pub mod opencl;
#[cfg(feature = "opencl")]
pub use opencl as game;

#[cfg(feature = "cuda")]
pub mod cuda;
#[cfg(feature = "cuda")]
pub use cuda as game;

#[cfg(feature = "simd")]
pub mod simd;
#[cfg(feature = "simd")]
pub use simd as game;
