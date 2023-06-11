use std::env;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=src/kernels/gol.cu");

    let out_dir = env::var("OUT_DIR").unwrap();

    let status = Command::new("nvcc")
        .args([
            "-O3",
            "-lib",
            "-gencode=arch=compute_61,code=sm_61",
            "src/kernels/gol.cu",
            "-o",
        ])
        .arg(&format!("{}/kernel.lib", &out_dir))
        .status()
        .unwrap();
    if !status.success() {
        panic!("Failed to compile kernel.");
    }

    println!("cargo:rustc-link-search=native={}", &out_dir);
}
