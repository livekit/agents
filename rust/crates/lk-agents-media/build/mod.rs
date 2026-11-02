//! Builds (or finds) the trimmed static ffmpeg and compiles `csrc/lk_agents_media.c` against it,
//! only under the `ffmpeg`/`blingfire` features. Sources, in order: `LK_CODECS_FFMPEG_PREFIX`
//! (prebuilt; the route for CI and Windows, where ffmpeg must be `--toolchain=msvc`), the pinned
//! `vendor/ffmpeg-<pin>/`, else `vendor/fetch.sh` unless `LK_CODECS_NO_FETCH=1`.
//! The configure line is LGPL-only: never add `--enable-gpl` or `--enable-nonfree`.

#![allow(
    clippy::disallowed_methods,
    clippy::disallowed_types,
    clippy::disallowed_macros,
    reason = "Build tooling reads native configuration; core purity applies to product code."
)]

mod blingfire;
mod cache;
mod libraries;
mod settings;

fn main() {
    println!("cargo:rerun-if-env-changed=LK_AGENTS_NATIVE_DEBUG");
    if std::env::var_os("CARGO_FEATURE_BLINGFIRE").is_some() {
        blingfire::build();
    }

    println!("cargo:rerun-if-changed=csrc/lk_agents_media.c");
    println!("cargo:rerun-if-changed=vendor/ffmpeg.pin");
    println!("cargo:rerun-if-env-changed=LK_CODECS_FFMPEG_PREFIX");
    println!("cargo:rustc-check-cfg=cfg(ffmpeg_built)");

    if std::env::var_os("CARGO_FEATURE_FFMPEG").is_none() {
        return;
    }

    let prefix = match std::env::var_os("LK_CODECS_FFMPEG_PREFIX") {
        Some(p) => std::path::PathBuf::from(p),
        None => build_vendored(),
    };

    let inc = prefix.join("include");
    let lib = prefix.join("lib");
    assert!(
        libraries::archive(&lib, "avcodec").is_some(),
        "no avcodec static library under {}; set LK_CODECS_FFMPEG_PREFIX or run vendor/fetch.sh",
        lib.display()
    );

    settings::compiler(false)
        .file("csrc/lk_agents_media.c")
        .include(&inc)
        .flag_if_supported("-std=c11")
        .warnings(true)
        .compile("lk_agents_media_shim");

    println!("cargo:rustc-link-search=native={}", lib.display());
    // Order matters for static archives: dependents before dependencies.
    for name in ["avformat", "avcodec", "swresample", "avutil"] {
        link_archive(&lib, name);
    }
    // Prebuilt installs can carry Opus themselves. Otherwise let pkg-config handle
    // target-specific configuration and platform linker flags.
    if libraries::archive(&lib, "opus").is_some() {
        link_archive(&lib, "opus");
    } else {
        pkg_config::Config::new().probe("opus")
            .expect("Opus is required: install its development package or include its static library in LK_CODECS_FFMPEG_PREFIX/lib");
    }
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        // libavutil's mach_absolute_time / VideoToolbox-free build still wants these.
        println!("cargo:rustc-link-lib=framework=CoreFoundation");
        println!("cargo:rustc-link-lib=framework=CoreMedia");
        println!("cargo:rustc-link-lib=framework=AudioToolbox");
        println!("cargo:rustc-link-lib=z");
        println!("cargo:rustc-link-lib=bz2");
        println!("cargo:rustc-link-lib=iconv");
    } else if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux") {
        println!("cargo:rustc-link-lib=m");
        println!("cargo:rustc-link-lib=z");
    }
    println!("cargo:rustc-cfg=ffmpeg_built");
}

/// The pinned configure line: `--disable-everything` plus an allow-list takes the build from
/// ~4,000 objects to 276.
const CONFIGURE: &[&str] = &[
    "--disable-everything",
    "--disable-programs",
    "--disable-doc",
    "--disable-network",
    "--disable-avdevice",
    "--disable-avfilter",
    "--disable-swscale",
    "--disable-postproc",
    "--enable-decoder=mp3,aac,flac,vorbis,opus,pcm_s16le,pcm_s16be,pcm_f32le",
    "--enable-demuxer=mp3,aac,flac,wav,ogg,matroska,mov",
    "--enable-parser=mpegaudio,aac,flac,opus,vorbis",
    "--enable-encoder=libopus",
    "--enable-muxer=ogg",
    "--enable-libopus",
    "--enable-static",
    "--disable-shared",
    "--enable-pic",
    // Stops configure picking up whatever the build machine has installed; needs the explicit
    // `--enable-libopus` above.
    "--disable-autodetect",
];

fn build_vendored() -> std::path::PathBuf {
    let manifest = std::path::PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let pin =
        std::fs::read_to_string(manifest.join("vendor/ffmpeg.pin")).expect("vendor/ffmpeg.pin");
    let version = pin
        .lines()
        .find_map(|l| l.trim().strip_prefix("version="))
        .expect("vendor/ffmpeg.pin has no version=")
        .to_owned();
    let src = manifest.join(format!("vendor/ffmpeg-{version}"));

    if !src.join("configure").exists() {
        assert!(
            std::env::var_os("LK_CODECS_NO_FETCH").is_none(),
            "vendor/ffmpeg-{version} is missing and LK_CODECS_NO_FETCH is set"
        );
        let st = std::process::Command::new("bash")
            .arg(manifest.join("vendor/fetch.sh"))
            .status()
            .expect("running vendor/fetch.sh");
        assert!(st.success(), "vendor/fetch.sh failed");
    }

    assert_eq!(
        std::env::var("HOST").expect("HOST"),
        std::env::var("TARGET").expect("TARGET"),
        "cross-compiling FFmpeg requires LK_CODECS_FFMPEG_PREFIX for the target"
    );
    println!("cargo:rerun-if-changed={}", src.display());
    let mut key = settings::key(false);
    key.input("recipe", include_bytes!("mod.rs"));
    key.tree(&src).expect("hash ffmpeg source tree");
    key.input(
        "x86asm",
        if which("nasm").is_some() || which("yasm").is_some() { b"enabled" } else { b"disabled" },
    );
    for args in
        [vec!["--version"], vec!["--modversion", "opus"], vec!["--cflags", "--libs", "opus"]]
    {
        if let Ok(output) = std::process::Command::new("pkg-config").args(args).output() {
            key.input("pkg-config-stdout", &output.stdout);
            key.input("pkg-config-stderr", &output.stderr);
        }
    }
    cache::get_or_build(
        &settings::root(),
        "ffmpeg",
        &key.finish(),
        |prefix| {
            prefix.join("include/libavcodec/avcodec.h").is_file()
                && ["avcodec", "avformat", "avutil", "swresample"]
                    .iter()
                    .all(|name| libraries::archive(&prefix.join("lib"), name).is_some())
        },
        |staging, prefix| {
            compile_ffmpeg(&src, staging, prefix);
            Ok(())
        },
    )
    .expect("build or reuse cached ffmpeg")
}

fn compile_ffmpeg(src: &std::path::Path, entry: &std::path::Path, prefix: &std::path::Path) {
    let build = entry.join("build");
    std::fs::create_dir_all(&build).expect("create ffmpeg build directory");

    let mut args: Vec<String> = vec![format!("--prefix={}", prefix.display())];
    args.extend(CONFIGURE.iter().map(|s| (*s).to_owned()));
    // Apple Clang still defaults to common symbols. FFmpeg's large tables then make
    // ld over-align __common beyond the segment limit; use real BSS definitions instead.
    if std::env::var("TARGET").is_ok_and(|target| target.contains("-apple-")) {
        args.push("--extra-cflags=-fno-common".into());
    }
    if !settings::native_debug() {
        args.push("--disable-debug".into());
    }
    // Without nasm configure aborts rather than degrading, so degrade explicitly.
    if which("nasm").is_none() && which("yasm").is_none() {
        args.push("--disable-x86asm".into());
    }

    run(
        std::process::Command::new("bash")
            .arg(src.join("configure"))
            .args(&args)
            .current_dir(&build),
        "configure",
    );
    let jobs = std::thread::available_parallelism().map_or(4, |n| n.get());
    run(std::process::Command::new("make").arg(format!("-j{jobs}")).current_dir(&build), "make");
    // DESTDIR isolates an interrupted installer from the final cache entry while
    // keeping the final prefix embedded in FFmpeg's pkg-config files.
    let destination = entry.join("install");
    run(
        std::process::Command::new("make")
            .arg("install")
            .arg(format!("DESTDIR={}", destination.display()))
            .current_dir(&build),
        "make install",
    );
    let relative_prefix: std::path::PathBuf =
        prefix
            .components()
            .filter_map(|part| {
                if let std::path::Component::Normal(name) = part { Some(name) } else { None }
            })
            .collect();
    let installed = destination.join(relative_prefix);
    for directory in ["include", "lib"] {
        std::fs::rename(installed.join(directory), entry.join(directory))
            .expect("collect installed FFmpeg artifacts");
    }
    std::fs::remove_dir_all(build).expect("remove ffmpeg build intermediates");
    std::fs::remove_dir_all(destination).expect("remove FFmpeg installation staging directory");
}

/// MSVC accepts exact filenames; Unix linkers use their conventional archive names.
fn link_archive(directory: &std::path::Path, name: &str) {
    let path = libraries::archive(directory, name)
        .unwrap_or_else(|| panic!("missing static {name} library under {}", directory.display()));
    if std::env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc") {
        let filename =
            path.file_name().expect("archive filename").to_str().expect("ASCII library name");
        println!("cargo:rustc-link-lib=static:+verbatim={filename}");
    } else {
        println!("cargo:rustc-link-lib=static={name}");
    }
}

fn which(bin: &str) -> Option<std::path::PathBuf> {
    std::env::var_os("PATH")
        .and_then(|paths| std::env::split_paths(&paths).map(|d| d.join(bin)).find(|p| p.is_file()))
}

fn run(cmd: &mut std::process::Command, what: &str) {
    let st = cmd.status().unwrap_or_else(|e| panic!("spawning {what}: {e}"));
    assert!(st.success(), "{what} failed with {st}");
}
