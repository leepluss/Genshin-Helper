use std::ffi::OsStr;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    if std::env::var_os("CARGO_CFG_TARGET_OS").as_deref() == Some(OsStr::new("windows")) {
        let mut res = winresource::WindowsResource::new();

        let icon_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("..")
            .join("assets")
            .join("icons")
            .join("app.ico");
        if icon_path.is_file() {
            println!("cargo:rerun-if-changed={}", icon_path.display());
            res.set_icon(&icon_path.to_string_lossy());
        }

        res.set("ProductName", "Genshin Helper");
        res.set(
            "FileDescription",
            "Genshin Helper - Exploration & HoYoLAB Auto-Sync",
        );
        res.set("OriginalFilename", "GenshinHelper.exe");
        res.set("InternalName", "GenshinHelper");

        if let Err(err) = res.compile() {
            panic!("failed to compile Windows resource: {err}");
        }
    }
}
