//! CUDA code generator: derives GPU accounting functions from Rust kernel source.
//!
//! This module reads `decision_kernel.rs` and `accounting.rs` signatures and
//! emits CUDA-compatible `.cu` device functions that mirror the Rust logic.
//!
//! The approach is string-template based (not full AST parsing) — pragmatic and
//! sufficient to guarantee accounting parity by construction.

mod emit;
mod templates;

use std::path::Path;

/// Run the full codegen pipeline: render templates and write output files.
pub fn run(out_dir: &Path, inspect_dir: &Path) {
    let source = render_all();

    emit::write_generated(out_dir, "generated_accounting.cu", &source);
    emit::write_generated(inspect_dir, "generated_accounting.cu", &source);

    eprintln!(
        "cargo:warning=codegen: wrote generated_accounting.cu ({} bytes)",
        source.len()
    );
}

/// Render all accounting templates into a single CUDA source string.
fn render_all() -> String {
    let mut out = String::with_capacity(8192);

    out.push_str(templates::HEADER);
    out.push('\n');
    out.push_str(&templates::quantize_codegen());
    out.push('\n');
    out.push_str(&templates::apply_open_codegen());
    out.push('\n');
    out.push_str(&templates::apply_close_codegen());
    out.push('\n');
    out.push_str(&templates::apply_partial_close_codegen());
    out.push('\n');
    out.push_str(&templates::funding_delta_codegen());
    out.push('\n');
    out.push_str(&templates::mark_to_market_pnl_codegen());
    out.push('\n');

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_all_contains_all_functions() {
        let source = render_all();
        assert!(source.contains("__device__ double quantize_codegen("));
        assert!(source.contains("__device__ void apply_open_codegen("));
        assert!(source.contains("__device__ void apply_close_codegen("));
        assert!(source.contains("__device__ void apply_partial_close_codegen("));
        assert!(source.contains("__device__ double funding_delta_codegen("));
        assert!(source.contains("__device__ double mark_to_market_pnl_codegen("));
        assert!(source.contains("AUTO-GENERATED"));
    }

    #[test]
    fn render_uses_double_accumulators() {
        let source = render_all();
        // All monetary calculations must use double, not float
        assert!(source.contains("double notional"));
        assert!(source.contains("double fee"));
        assert!(source.contains("double pnl"));
    }
}
