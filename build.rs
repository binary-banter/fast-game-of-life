use std::process::Command;

fn main() {
    // Tell Cargo that if the given file changes, to rerun this build script.
    println!("cargo:rerun-if-changed=src/kernels/gol.cu");
    if let Ok(mut p) = Command::new("nvcc")
        .arg("-ptx")
        .arg("-gencode=arch=compute_61,code=sm_61")
        .arg("./src/kernels/gol.cu")
        .arg("-o")
        .arg("./target/gol.ptx")
        .spawn() {
        let exit = p.wait().unwrap();
        if !exit.success() {
            panic!("Failed to build kernel.");
        }
    }
}