#![cfg_attr(feature = "simd", feature(portable_simd))]
#![cfg_attr(feature = "simd", feature(array_windows))]
#![cfg_attr(feature = "simd", feature(array_chunks))]

#[cfg(all(feature = "opencl", feature = "cuda"))]
compile_error!("The \"opencl\" and \"cuda\" features are mutually exclusive!");

#[cfg(all(feature = "opencl", feature = "simd"))]
compile_error!("The \"opencl\" and \"simd\" features are mutually exclusive!");

#[cfg(all(feature = "cuda", feature = "simd"))]
compile_error!("The \"cuda\" and \"simd\" features are mutually exclusive!");

#[cfg(feature = "cuda")]
pub mod cuda;
#[cfg(feature = "opencl")]
pub mod opencl;
#[cfg(feature = "simd")]
pub mod simd;
pub mod trivial;
