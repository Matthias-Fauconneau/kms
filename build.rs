fn main() {
    println!("cargo:rerun-if-changed=va.h");
    bindgen::builder().header("va.h").allowlist_function("va.*").allowlist_type("VA.*").allowlist_var("VA.*").parse_callbacks(Box::new(bindgen::CargoCallbacks)).layout_tests(false).generate_comments(false).derive_default(true).derive_debug(true).generate().unwrap()
        .write_to_file(std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap()).join("va.rs")).unwrap();
    println!("cargo:rustc-link-lib=va");
    println!("cargo:rustc-link-lib=va-drm");
}