fn main() {
    /*let files = std::fs::read_dir("libavutil").unwrap().filter_map(|res| res.map(|e| e.file_type().unwrap().is_file().then(|| e.path())).unwrap()) .chain(
        std::fs::read_dir("libavcodec").unwrap().map(|res| res.map(|e| e.path()).unwrap()) ).chain(
        std::fs::read_dir("libavformat").unwrap().map(|res| res.map(|e| e.path()).unwrap())).filter(|path| path.ends_with(".c")).collect::<Vec<_>>();
    //eprintln!("{:?}", &files);
    cc::Build::new().includes([".","/usr/include/drm"]).define("HAVE_AV_CONFIG_H","").extra_warnings(false).files(files).compile("av");*/

    println!("cargo:rerun-if-changed=av.h");
    bindgen::builder().clang_arg("-I.").header("av.h").allowlist_function("av.*").allowlist_var("AV.*").parse_callbacks(Box::new(bindgen::CargoCallbacks))/*.derive_default(true)*/.layout_tests(false).generate_comments(false).generate().unwrap()
        .write_to_file(std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap()).join("av.rs")).unwrap();
    println!("cargo:rustc-link-search=.");
    println!("cargo:rustc-link-lib=av");
    /*println!("cargo:rustc-link-lib=avcodec");
    println!("cargo:rustc-link-lib=avformat");
    println!("cargo:rustc-link-lib=avutil");*/

    println!("cargo:rerun-if-changed=va.h");
    bindgen::builder().header("va.h").allowlist_function("va.*").allowlist_type("VA.*").allowlist_var("VA.*").parse_callbacks(Box::new(bindgen::CargoCallbacks)).layout_tests(false).generate_comments(false).derive_default(true).derive_debug(true).generate().unwrap()
        .write_to_file(std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap()).join("va.rs")).unwrap();
    println!("cargo:rustc-link-lib=va");
    println!("cargo:rustc-link-lib=va-drm");
}