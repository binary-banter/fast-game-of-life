fn main() {
    #[cfg(feature = "opencl")]
    set_constants_opencl();

    #[cfg(feature = "cuda")]
    set_constants_cuda();
    #[cfg(feature = "cuda")]
    compile_cuda();
}

#[cfg(feature = "opencl")]
fn set_constants_opencl() {
    println!("cargo:rerun-if-changed=settings.toml");

    use std::fs::File;
    use std::io::Write;
    use toml::Table;

    let settings = include_str!("settings.toml").parse::<Table>().unwrap();
    let work_group_size = settings["opencl"]["work-group-size"].as_integer().unwrap();
    let work_per_thread = settings["opencl"]["work-per-thread"].as_integer().unwrap();
    let padding_x = settings["opencl"]["padding-x"].as_integer().unwrap();
    let padding_y = settings["opencl"]["padding-y"].as_integer().unwrap();

    let mut constants_file = File::create("./src/kernels/constants.h").unwrap();
    writeln!(constants_file, "#define WORK_GROUP_SIZE  {work_group_size}").unwrap();
    writeln!(constants_file, "#define WORK_PER_THREAD  {work_per_thread}").unwrap();
    writeln!(constants_file, "#define PADDING_X        {padding_x}").unwrap();
    writeln!(constants_file, "#define PADDING_Y        {padding_y}").unwrap();
}

#[cfg(feature = "cuda")]
fn set_constants_cuda() {
    println!("cargo:rerun-if-changed=settings.toml");

    use std::fs::File;
    use std::io::Write;
    use toml::Table;

    let settings = include_str!("settings.toml").parse::<Table>().unwrap();
    let work_per_thread = settings["cuda"]["work-per-thread"].as_integer().unwrap();
    let step_size = settings["cuda"]["step-size"].as_integer().unwrap();

    let mut constants_file = File::create("./src/kernels/constants.h").unwrap();
    writeln!(constants_file, "#define WORK_PER_THREAD {work_per_thread}").unwrap();
    writeln!(constants_file, "#define STEP_SIZE {step_size}").unwrap();
}

#[cfg(feature = "cuda")]
fn compile_cuda() {
    println!("cargo:rerun-if-changed=src/kernels/gol.cu");

    let out_dir = std::env::var("OUT_DIR").unwrap();

    let filename = if cfg!(target_os = "windows") {
        "kernel.lib"
    } else if cfg!(target_os = "linux") {
        "libkernel.a"
    } else {
        panic!("Unsupported distribution.");
    };

    let status = std::process::Command::new("nvcc")
        .args([
            "-O3",
            "--compiler-options",
            "-fpie",
            "-lib",
            "--gpu-architecture=native",
            "src/kernels/gol.cu",
            "-o",
        ])
        .arg(&format!("{out_dir}/{filename}"))
        .status()
        .unwrap();

    if !status.success() {
        panic!("Failed to compile kernel.");
    }

    println!("cargo:rustc-link-search=native={}", &out_dir);
}
