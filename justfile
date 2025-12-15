# justfile - lives in agents/
#
# Common workflows:
#   just link-python-rtc              # Link to local python-rtc (pre-built rust artifacts)
#   just link-rust-sdk                # Build and link local rust SDK from source
#   just unlink                       # Restore PyPI version
#   just status                       # Show current linking status
#   just run <command>                # Run command (auto-uses local rust if active)
#
# Cross-compilation (override auto-detection):
#   just --set _detected_os linux --set _detected_arch x86_64 link-python-rtc
#   just --set _detected_os android --set _detected_arch arm64 link-python-rtc
#   just --set _detected_os ios --set _detected_arch arm64 link-python-rtc
#
# Available platforms: macos, linux, windows, android, ios, ios-sim
# Available architectures: arm64, x86_64, armv7 (android only)

set dotenv-load

# Paths (relative to agents/)
agents_project := "livekit-agents"
python_rtc := "../python-sdks/livekit-rtc"
rust_sdks := "../rust-sdks"
rust_submodule := "../python-sdks/livekit-rtc/rust-sdks"

# Package name as it appears in pip
package_name := "livekit"

# Platform and architecture auto-detection
_detected_arch := arch()
_detected_os := os()

# Default recipe
default:
    @just --list

# ============================================
# Main development workflows
# ============================================

# Link to local python-rtc (downloads pre-built rust artifacts)
link-python-rtc:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "🐍 Linking local python-rtc..."

    # Auto-detect platform and arch
    DETECTED_ARCH="{{ _detected_arch }}"
    DETECTED_OS="{{ _detected_os }}"

    # Normalize arch (aarch64 -> arm64)
    if [ "$DETECTED_ARCH" = "aarch64" ]; then
        ARCH="arm64"
    else
        ARCH="$DETECTED_ARCH"
    fi

    # Normalize platform
    if [ "$DETECTED_OS" = "macos" ]; then
        PLATFORM="macos"
    elif [ "$DETECTED_OS" = "linux" ]; then
        PLATFORM="linux"
    elif [ "$DETECTED_OS" = "windows" ]; then
        PLATFORM="windows"
    else
        PLATFORM="$DETECTED_OS"
    fi

    # Download pre-built FFI artifacts
    echo "📦 Downloading FFI artifacts for $PLATFORM-$ARCH..."
    cd "{{ python_rtc }}"
    python rust-sdks/download_ffi.py --platform "$PLATFORM" --arch "$ARCH" --output livekit/rtc/resources

    # Add local python-rtc to agents project
    echo "🔗 Adding local python-rtc to agents..."
    cd "{{ justfile_directory() }}/{{ agents_project }}"
    uv add --editable "../../python-sdks/livekit-rtc"
    uv sync

    # Clear any rust-sdk env var (using pre-built artifacts)
    rm -f .env

    echo "✅ Linked to local python-rtc (with downloaded rust artifacts)"

# Link to local rust-sdk (builds from source, implies local python-rtc)
link-rust-sdk: sync-rust-submodule build-rust generate-proto
    #!/usr/bin/env bash
    set -euo pipefail
    echo "🐍 Linking python-rtc with local rust build..."

    RUST_LIB_PATH="$(cd "{{ rust_sdks }}" && pwd)/target/release"
    echo "   LIVEKIT_LIB_PATH=$RUST_LIB_PATH"

    # Add local python-rtc to agents project
    echo "🔗 Adding local python-rtc to agents..."
    cd "{{ justfile_directory() }}/{{ agents_project }}"
    uv add --editable "../../python-sdks/livekit-rtc"
    uv sync

    # Save rust-sdk path to .env for automatic use
    echo "LIVEKIT_LIB_PATH=$RUST_LIB_PATH" > .env

    echo "✅ Linked to local rust-sdk + python-rtc"

# Link to a specific python-rtc version from PyPI
link-python-rtc-version version:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "📦 Installing livekit version {{ version }}..."
    
    cd "{{ justfile_directory() }}/{{ agents_project }}"
    uv remove --dev {{ package_name }} 2>/dev/null || true
    uv add "{{ package_name }}=={{ version }}"
    uv sync
    
    echo "✅ Linked to livekit {{ version }}"

# Unlink local and restore PyPI version
unlink:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "🔓 Unlinking local python-rtc..."

    # Remove the workspace root [tool.uv.sources] override for livekit
    cd "{{ justfile_directory() }}"
    if grep -q "^livekit = { path = " pyproject.toml; then
        echo "   Removing workspace root livekit source override..."
        sed -i.bak '/^livekit = { path = /d' pyproject.toml
        rm -f pyproject.toml.bak
    fi

    cd "{{ justfile_directory() }}/{{ agents_project }}"

    # Remove the [tool.uv.sources] override for livekit (if it exists)
    if grep -q "\[tool.uv.sources\]" pyproject.toml; then
        echo "   Removing [tool.uv.sources] override..."
        # Remove the [tool.uv.sources] section and the livekit source line
        sed -i.bak '/\[tool.uv.sources\]/,/^livekit = /d' pyproject.toml
        rm -f pyproject.toml.bak
    fi

    # Remove and re-add the package to get PyPI version
    uv remove --dev {{ package_name }} 2>/dev/null || true
    uv add --upgrade-package {{ package_name }} {{ package_name }}
    uv sync

    # Clear any rust-sdk env var
    rm -f .env

    echo "✅ Restored PyPI version"

# ============================================
# Helper recipes
# ============================================

# Sync rust-sdks submodule to match standalone rust-sdks
sync-rust-submodule:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "🔄 Syncing rust-sdks submodule to match standalone..."
    
    # Get the commit of standalone rust-sdks
    STANDALONE_COMMIT=$(cd "{{ rust_sdks }}" && git rev-parse HEAD)
    echo "   Standalone rust-sdks at: $STANDALONE_COMMIT"
    
    # Update submodule to same commit
    cd "{{ rust_submodule }}"
    git fetch origin
    git checkout "$STANDALONE_COMMIT"
    
    echo "   Submodule synced to: $STANDALONE_COMMIT"

# Build livekit-ffi from rust-sdks
build-rust:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "🦀 Building livekit-ffi..."
    cd "{{ rust_sdks }}" && cargo build --release -p livekit-ffi
    echo "   Built: {{ rust_sdks }}/target/release"

# Generate protobuf FFI protocol
generate-proto:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "📝 Generating protobuf FFI protocol..."
    cd "{{ python_rtc }}" && ./generate_proto.sh
    echo "   Protos generated"

# ============================================
# Run commands
# ============================================

# Run a command (automatically uses local rust SDK if available)
run *args:
    #!/usr/bin/env bash
    set -euo pipefail
    cd "{{ justfile_directory() }}/{{ agents_project }}"

    # Load .env if it exists (contains LIVEKIT_LIB_PATH for local rust builds)
    if [ -f .env ]; then
        set -a
        source .env
        set +a
    fi

    uv run {{ args }}

# ============================================
# Utilities
# ============================================

# Show current linking status
status:
    #!/usr/bin/env bash
    echo "📍 Current status:"
    echo ""

    # Store absolute paths before changing directory
    RUST_SDKS_DIR="$(cd "{{ justfile_directory() }}/{{ rust_sdks }}" 2>/dev/null && pwd || echo "")"
    RUST_SUBMODULE_DIR="$(cd "{{ justfile_directory() }}/{{ rust_submodule }}" 2>/dev/null && pwd || echo "")"

    cd "{{ justfile_directory() }}/{{ agents_project }}"

    # Check if livekit is installed and how
    SHOW_OUTPUT=$(uv pip show livekit 2>/dev/null || echo "")

    if [ -z "$SHOW_OUTPUT" ]; then
        echo "   livekit: NOT INSTALLED"
    elif echo "$SHOW_OUTPUT" | grep -q "Editable project location:"; then
        VERSION=$(echo "$SHOW_OUTPUT" | grep "^Version:" | awk '{print $2}')
        LOCATION=$(echo "$SHOW_OUTPUT" | grep "Editable project location:" | cut -d' ' -f4-)
        echo "   livekit: LOCAL (editable) v$VERSION"
        echo "   path:    $LOCATION"
    else
        VERSION=$(echo "$SHOW_OUTPUT" | grep "^Version:" | awk '{print $2}')
        echo "   livekit: PyPI ($VERSION)"
    fi

    echo ""

    # Check rust builds using absolute paths
    RUST_ACTIVE="(NOT ACTIVE)"
    if [ -f .env ] && grep -q "LIVEKIT_LIB_PATH" .env; then
        RUST_ACTIVE=" (ACTIVE)"
    fi

    if [ -n "$RUST_SDKS_DIR" ] && { [ -f "$RUST_SDKS_DIR/target/release/liblivekit_ffi.dylib" ] || \
       [ -f "$RUST_SDKS_DIR/target/release/liblivekit_ffi.so" ]; }; then
        echo "   rust-sdk: LOCAL BUILD AVAILABLE$RUST_ACTIVE"
    else
        echo "   rust-sdk: no local build"
    fi

    echo ""
    echo "   Standalone rust-sdks: $([ -n "$RUST_SDKS_DIR" ] && cd "$RUST_SDKS_DIR" 2>/dev/null && git rev-parse --short HEAD || echo 'N/A')"
    echo "   Submodule rust-sdks:  $([ -n "$RUST_SUBMODULE_DIR" ] && cd "$RUST_SUBMODULE_DIR" 2>/dev/null && git rev-parse --short HEAD || echo 'N/A')"

# Clean builds and reset to PyPI dependencies
clean:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "🧹 Cleaning..."

    # Remove local editable and restore PyPI
    cd "{{ justfile_directory() }}/{{ agents_project }}"
    uv remove --dev {{ package_name }} 2>/dev/null || true
    uv add {{ package_name }}
    uv sync

    # Clear any rust-sdk env var
    rm -f .env

    # Clean rust build
    cd "{{ justfile_directory() }}"
    cd "{{ rust_sdks }}" && cargo clean

    echo "   Cleaned!"

# Show paths for debugging
info:
    #!/usr/bin/env bash
    # Auto-detect platform and arch
    DETECTED_ARCH="{{ _detected_arch }}"
    DETECTED_OS="{{ _detected_os }}"

    # Normalize arch (aarch64 -> arm64)
    if [ "$DETECTED_ARCH" = "aarch64" ]; then
        ARCH="arm64"
    else
        ARCH="$DETECTED_ARCH"
    fi

    # Normalize platform
    if [ "$DETECTED_OS" = "macos" ]; then
        PLATFORM="macos"
    elif [ "$DETECTED_OS" = "linux" ]; then
        PLATFORM="linux"
    elif [ "$DETECTED_OS" = "windows" ]; then
        PLATFORM="windows"
    else
        PLATFORM="$DETECTED_OS"
    fi

    echo "agents-project: {{ agents_project }}"
    echo "python-rtc:     {{ python_rtc }}"
    echo "rust-sdks:      {{ rust_sdks }}"
    echo "rust-submodule: {{ rust_submodule }}"
    echo "package-name:   {{ package_name }}"
    echo "platform:       $PLATFORM (detected: $DETECTED_OS)"
    echo "arch:           $ARCH (detected: $DETECTED_ARCH)"