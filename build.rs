#[cfg(any(feature = "cuda", feature = "opencl"))]
fn main() {
    set_constants();
    if cfg!(feature = "cuda") {
        compile_cuda();
    }
}

#[cfg(not(any(feature = "cuda", feature = "opencl")))]
fn main() {}

#[cfg(any(feature = "cuda", feature = "opencl"))]
fn set_constants() {
    println!("cargo:rerun-if-changed=settings.toml");
    println!("cargo:rerun-if-changed=./src/kernels/constants.h");

    use std::fs::File;
    use std::io::Write;
    use toml::Table;

    let settings = include_str!("settings.toml").parse::<Table>().unwrap();

    let key = if cfg!(feature = "cuda") {
        "cuda"
    } else if cfg!(feature = "opencl") {
        "opencl"
    } else {
        unreachable!("This feature has no constants to be set!");
    };

    let work_group_size = settings[key]["work-group-size"].as_integer().unwrap();
    let work_per_thread = settings[key]["work-per-thread"].as_integer().unwrap();
    let padding_x = settings[key]["padding-x"].as_integer().unwrap();
    let padding_y = settings[key]["padding-y"].as_integer().unwrap();

    let mut constants_file = File::create("./src/kernels/constants.h").unwrap();
    writeln!(constants_file, "#define WORK_GROUP_SIZE  {work_group_size}").unwrap();
    writeln!(constants_file, "#define WORK_PER_THREAD  {work_per_thread}").unwrap();
    writeln!(constants_file, "#define PADDING_X        {padding_x}").unwrap();
    writeln!(constants_file, "#define PADDING_Y        {padding_y}").unwrap();
}

#[cfg(any(feature = "cuda", feature = "opencl"))]
fn compile_cuda() {
    println!("cargo:rerun-if-changed=src/kernels/gol.cu");

    use std::env;
    use std::process::Command;

    let out_dir = env::var("OUT_DIR").unwrap();
    let filename = if cfg!(target_os = "windows") {
        "kernel.lib"
    } else if cfg!(target_os = "linux") {
        "libkernel.a"
    } else {
        panic!("Unsupported distribution.");
    };
    let status = Command::new("nvcc")
        .args([
            "-O3",
            "--compiler-options",
            "-fpie",
            "-lib",
            "-gencode=arch=compute_61,code=sm_61",
            "src/kernels/gol.cu",
            "-o",
        ])
        .arg(&format!("{}/{}", &out_dir, filename))
        .status()
        .unwrap();
    if !status.success() {
        panic!("Failed to compile kernel.");
    }
    println!("cargo:rustc-link-search=native={}", &out_dir);
}
