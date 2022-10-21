fn main() {
    println!("cargo:rustc-link-lib=va");
    println!("cargo:rustc-link-lib=va-drm");
    println!("cargo:rerun-if-changed=wrapper.h");
    bindgen::Builder::default().header("wrapper.h").parse_callbacks(Box::new(bindgen::CargoCallbacks)).derive_default(true).generate().unwrap()
        .write_to_file(std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap()).join("bindings.rs")).unwrap();
    cc::Build::new().files(std::fs::read_dir("libavutil").unwrap().map(|res| res.map(|e| e.path()).unwrap()));
    cc::Build::new().files(std::fs::read_dir("libavcodec").unwrap().map(|res| res.map(|e| e.path()).unwrap()));
    cc::Build::new().files(std::fs::read_dir("libavformat").unwrap().map(|res| res.map(|e| e.path()).unwrap()));
}