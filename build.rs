use std::env;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=src/kernels/gol.cu");

    let out_dir = env::var("OUT_DIR").unwrap();

    Command::new("nvcc")
        .args(&["-O3",
            "src/kernels/gol.cu",
            "-lib",
            "-ccbin",
            "cl.exe",
            "-Xcompiler", "-wd4819",
            "-o",
        ])
        .arg(&format!("{}/kernel.lib", &out_dir))
        .status()
        .unwrap();

    println!("cargo:rustc-link-search=native={}", &out_dir);
}