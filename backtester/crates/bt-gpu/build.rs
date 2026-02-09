fn main() {
    println!("cargo:rerun-if-changed=kernels/sweep_engine.cu");
    println!("cargo:rerun-if-changed=kernels/indicator_kernel.cu");

    let out_dir = std::env::var("OUT_DIR").unwrap();

    // Compile sweep_engine.cu → PTX
    let ptx_sweep = format!("{}/sweep_engine.ptx", out_dir);
    let status = std::process::Command::new("nvcc")
        .args(&[
            "--ptx",
            "-arch=sm_86", // RTX 3090 = Ampere
            "-O3",
            "-o",
            &ptx_sweep,
            "kernels/sweep_engine.cu",
        ])
        .status()
        .expect("Failed to run nvcc. Is CUDA toolkit installed?");
    if !status.success() {
        panic!("nvcc failed to compile sweep_engine.cu");
    }

    // Compile indicator_kernel.cu → PTX
    let ptx_indicator = format!("{}/indicator_kernel.ptx", out_dir);
    let status = std::process::Command::new("nvcc")
        .args(&[
            "--ptx",
            "-arch=sm_86",
            "-O3",
            "-o",
            &ptx_indicator,
            "kernels/indicator_kernel.cu",
        ])
        .status()
        .expect("Failed to run nvcc for indicator_kernel.cu");
    if !status.success() {
        panic!("nvcc failed to compile indicator_kernel.cu");
    }
}
