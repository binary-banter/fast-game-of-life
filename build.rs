#[cfg(feature = "cuda")]
fn main() {
    use std::env;
    use std::process::Command;
    println!("cargo:rerun-if-changed=src/kernels/gol.cu");

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
            "--compiler-options", "-fpie",
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

#[cfg(not(feature = "cuda"))]
fn main() {}
