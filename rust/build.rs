//! Build script: generates `phoenixdb.h` from the `extern "C"` surface via cbindgen.
//!
//! The header is emitted to two places:
//!   1. `$OUT_DIR/phoenixdb.h`            (always, for downstream `cc` consumers)
//!   2. `<repo>/native/include/phoenixdb.h` (checked into the Dart package)

use std::env;
use std::path::PathBuf;

fn main() {
    let crate_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));

    println!("cargo:rerun-if-changed=src/ffi.rs");
    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=cbindgen.toml");
    println!("cargo:rerun-if-env-changed=PHOENIXDB_SKIP_CBINDGEN");

    if env::var_os("PHOENIXDB_SKIP_CBINDGEN").is_some() {
        println!("cargo:warning=cbindgen skipped (PHOENIXDB_SKIP_CBINDGEN set)");
        return;
    }

    let config = cbindgen::Config::from_file(crate_dir.join("cbindgen.toml"))
        .expect("failed to read cbindgen.toml");

    let bindings = match cbindgen::Builder::new()
        .with_crate(&crate_dir)
        .with_config(config)
        .generate()
    {
        Ok(b) => b,
        Err(e) => {
            // Never hard-fail the build of the library itself because of header
            // generation; surface it loudly instead.
            println!("cargo:warning=cbindgen failed: {e}");
            return;
        }
    };

    bindings.write_to_file(out_dir.join("phoenixdb.h"));

    // <repo>/native/include/phoenixdb.h  (crate lives at <repo>/rust)
    let repo_root = crate_dir.parent().unwrap_or(&crate_dir).to_path_buf();
    let include_dir = repo_root.join("native").join("include");
    if std::fs::create_dir_all(&include_dir).is_ok() {
        bindings.write_to_file(include_dir.join("phoenixdb.h"));
    }
}
