//! Builds (or finds) the trimmed static ffmpeg and compiles `csrc/lk_agents_media.c` against it,
//! only under the `ffmpeg`/`blingfire` features. Sources, in order: `LK_CODECS_FFMPEG_PREFIX`
//! (prebuilt; the route for CI and Windows, where ffmpeg must be `--toolchain=msvc`), the pinned
//! `vendor/ffmpeg-<pin>/`, else `vendor/fetch.sh` unless `LK_CODECS_NO_FETCH=1`.
//! The configure line is LGPL-only: never add `--enable-gpl` or `--enable-nonfree`.

#[path = "build/blingfire.rs"]
mod blingfire;

fn main() {
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
        find_lib(&lib, "avcodec"),
        "no avcodec static library under {}; set LK_CODECS_FFMPEG_PREFIX or run vendor/fetch.sh",
        lib.display()
    );

    cc::Build::new()
        .file("csrc/lk_agents_media.c")
        .include(&inc)
        .flag_if_supported("-std=c11")
        .warnings(true)
        .compile("lk_agents_media_shim");

    println!("cargo:rustc-link-search=native={}", lib.display());
    // Order matters for static archives: dependents before dependencies.
    for l in ["avformat", "avcodec", "swresample", "avutil"] {
        println!("cargo:rustc-link-lib=static={l}");
    }
    // libopus: pkg-config if it is there, otherwise assume the prefix carries it statically.
    if let Ok(out) = std::process::Command::new("pkg-config").args(["--libs", "opus"]).output()
        && out.status.success()
    {
        for tok in String::from_utf8_lossy(&out.stdout).split_whitespace() {
            if let Some(rest) = tok.strip_prefix("-L") {
                println!("cargo:rustc-link-search=native={rest}");
            } else if let Some(rest) = tok.strip_prefix("-l") {
                println!("cargo:rustc-link-lib={rest}");
            }
        }
    }
    if cfg!(target_os = "macos") {
        // libavutil's mach_absolute_time / VideoToolbox-free build still wants these.
        println!("cargo:rustc-link-lib=framework=CoreFoundation");
        println!("cargo:rustc-link-lib=framework=CoreMedia");
        println!("cargo:rustc-link-lib=framework=AudioToolbox");
        println!("cargo:rustc-link-lib=z");
        println!("cargo:rustc-link-lib=bz2");
        println!("cargo:rustc-link-lib=iconv");
    } else if cfg!(target_os = "linux") {
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

    let out = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let prefix = out.join("ffmpeg-prefix");
    if find_lib(&prefix.join("lib"), "avcodec") {
        return prefix;
    }
    let build = out.join("ffmpeg-build");
    std::fs::create_dir_all(&build).unwrap();

    let mut args: Vec<String> = vec![format!("--prefix={}", prefix.display())];
    args.extend(CONFIGURE.iter().map(|s| (*s).to_owned()));
    // Without nasm configure aborts rather than degrading, so degrade explicitly.
    if which("nasm").is_none() && which("yasm").is_none() {
        println!(
            "cargo:warning=neither nasm nor yasm found; building ffmpeg with --disable-x86asm"
        );
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
    run(std::process::Command::new("make").arg("install").current_dir(&build), "make install");
    prefix
}

/// `--toolchain=msvc` emits `libfoo.a` while an import-library build emits `foo.lib`; accept either.
fn find_lib(dir: &std::path::Path, stem: &str) -> bool {
    dir.join(format!("lib{stem}.a")).exists() || dir.join(format!("{stem}.lib")).exists()
}

fn which(bin: &str) -> Option<std::path::PathBuf> {
    std::env::var_os("PATH")
        .and_then(|paths| std::env::split_paths(&paths).map(|d| d.join(bin)).find(|p| p.is_file()))
}

fn run(cmd: &mut std::process::Command, what: &str) {
    let st = cmd.status().unwrap_or_else(|e| panic!("spawning {what}: {e}"));
    assert!(st.success(), "{what} failed with {st}");
}
