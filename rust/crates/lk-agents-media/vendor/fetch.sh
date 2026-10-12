#!/usr/bin/env bash
# Fetch and verify the pinned ffmpeg source into vendor/ffmpeg-<version>/.
#
# WHY THIS IS A SCRIPT AND NOT A COMMITTED TREE
#
# The extracted tree is 100 MB across 8,540 files. Committing it would dominate the repository
# and every clone of it, and a `git log` over it is unreadable. So the *pin* is committed
# (vendor/ffmpeg.pin: a URL and a sha256) and the bytes are fetched once. The sha256 is what
# makes this reproducible; the tarball is not trusted, it is verified.
#
# It stays INSIDE rust/crates/lk-agents-media/ deliberately: an out-of-workspace `path` dependency or
# a tree above the crate root makes `cargo-mutants` unusable workspace-wide, because it copies
# the tree to a temp dir and the relative path dangles.
#
# If your build environment has no network, either commit the extracted tree at the same path or
# set LK_CODECS_FFMPEG_PREFIX to a prebuilt static install. build.rs honours both.
set -euo pipefail
cd "$(dirname "$0")"

# shellcheck disable=SC2046
eval $(grep -E '^(version|url|sha256)=' ffmpeg.pin)
dir="ffmpeg-${version}"
tar="ffmpeg-${version}.tar.xz"

if [[ -f "$dir/configure" ]]; then
  echo "vendor/$dir already present"
  exit 0
fi

if [[ ! -f "$tar" ]]; then
  echo "fetching $url"
  curl -fsSL -o "$tar.part" "$url"
  mv "$tar.part" "$tar"
fi

got=$(shasum -a 256 "$tar" 2>/dev/null | awk '{print $1}' || sha256sum "$tar" | awk '{print $1}')
if [[ "$got" != "$sha256" ]]; then
  echo "FAIL: sha256 mismatch for $tar" >&2
  echo "  expected $sha256" >&2
  echo "  got      $got" >&2
  rm -f "$tar"
  exit 1
fi

rm -rf "$dir.tmp"
mkdir -p "$dir.tmp"
tar xf "$tar" -C "$dir.tmp" --strip-components=1
mv "$dir.tmp" "$dir"
echo "vendor/$dir ready"
