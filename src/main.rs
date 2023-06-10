mod game;

use rust_gpu_tools::{opencl, program_closures, Device, GPUError, Program};

#[cfg(feature = "cuda")]
use rust_gpu_tools::cuda;

pub fn main() {
    // Define some data that should be operated on.
    let aa: Vec<u32> = vec![1, 2, 3, 4];
    let bb: Vec<u32> = vec![5, 6, 7, 8];

    let closures = program_closures!(|program, _args| -> Result<Vec<u32>, GPUError> {
        // Make sure the input data has the same length.
        assert_eq!(aa.len(), bb.len());
        let length = aa.len();

        // Copy the data to the GPU.
        let aa_buffer = program.create_buffer_from_slice(&aa)?;
        let bb_buffer = program.create_buffer_from_slice(&bb)?;

        // The result buffer has the same length as the input buffers.
        let result_buffer = unsafe { program.create_buffer::<u32>(length)? };

        // Get the kernel.
        let kernel = program.create_kernel("add", 1, 1)?;

        // Execute the kernel.
        kernel
            .arg(&(length as u32))
            .arg(&aa_buffer)
            .arg(&bb_buffer)
            .arg(&result_buffer)
            .run()?;

        // Get the resulting data.
        let mut result = vec![0u32; length];
        program.read_into_buffer(&result_buffer, &mut result)?;

        Ok(result)
    });

    let device = Device::all()[0];
    let program = get_program(device);
    let result = program.run(closures, ()).unwrap();
    assert_eq!(result, [6, 8, 10, 12]);
    println!("Result: {:?}", result);
}

#[cfg(feature = "cuda")]
fn get_program(device: &Device) -> Program {
    // The kernel was compiled with:
    let cuda_kernel = CString::new("../target/gol.fatbin").unwrap();
    let cuda_device = device.cuda_device().unwrap();
    let cuda_program = cuda::Program::from_binary(cuda_device, &cuda_kernel).unwrap();
    Program::Cuda(cuda_program)
}

#[cfg(not(feature = "cuda"))]
fn get_program(device: &Device) -> Program {
    let opencl_kernel = include_str!("./kernels/gol.cl");
    let opencl_device = device.opencl_device().unwrap();
    let opencl_program = opencl::Program::from_opencl(opencl_device, opencl_kernel).unwrap();
    Program::Opencl(opencl_program)
}