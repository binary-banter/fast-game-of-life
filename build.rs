use std::process::Command;

// Example custom build script.
fn main() {
    // Tell Cargo that if the given file changes, to rerun this build script.
    println!("cargo:rerun-if-changed=src/kernels/gol.cu");
    Command::new("nvcc")
        .arg("-fatbin")
        .arg("-gencode=arch=compute_61,code=sm_61")
        .arg("./src/kernels/gol.cu")
        .arg("-o")
        .arg("./target/gol.fatbin")
        .spawn()
        .unwrap();
}
