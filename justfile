# justfile - lives in agents/
#
# Common workflows:
#   just link-rtc                     # Link to local python-rtc (default branch FFI version)
#   just link-rtc local               # Build and link local rust SDK from source
#   just link-rtc 0.18.0              # Link to specific livekit PyPI version
#   just unlink                       # Restore PyPI version
#   just status                       # Show current linking status
#   just run <command>                # Run command (auto-uses local rust if active)
#
# Cross-compilation (override auto-detection):
#   just --set _detected_os linux --set _detected_arch x86_64 link-rtc
#   just --set _detected_os android --set _detected_arch arm64 link-rtc
#   just --set _detected_os ios --set _detected_arch arm64 link-rtc
#
# Available platforms: macos, linux, windows, android, ios, ios-sim
# Available architectures: arm64, x86_64, armv7 (android only)

set dotenv-load

# Paths (relative to agents/)
agents_project := "livekit-agents"
python_rtc := "../python-sdks/livekit-rtc"
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

# Link to local python-rtc with optional version or 'local' for building from source
link-rtc version="": (_link-rtc-impl version)

# Internal implementation for link-rtc
_link-rtc-impl version:
    #!/usr/bin/env bash
    set -euo pipefail

    VERSION="{{ version }}"

    # Case 1: "local" - build and link local rust SDK from source
    if [ "$VERSION" = "local" ]; then
        echo "🦀 Building and linking local rust SDK..."
        just build-rust
        just generate-proto

        RUST_LIB_PATH="$(cd "{{ rust_submodule }}" && pwd)/target/release"
        echo "   LIVEKIT_LIB_PATH=$RUST_LIB_PATH"

        # Add local python-rtc to agents project
        echo "🔗 Adding local python-rtc to agents..."
        cd "{{ justfile_directory() }}/{{ agents_project }}"
        uv add --editable "../../python-sdks/livekit-rtc"
        uv sync

        # Save rust-sdk path to .env for automatic use
        echo "LIVEKIT_LIB_PATH=$RUST_LIB_PATH" > .env

        echo "✅ Linked to local rust-sdk + python-rtc"

    # Case 2: Version number specified - install from PyPI
    elif [ -n "$VERSION" ]; then
        echo "📦 Installing livekit version $VERSION..."

        # Remove the workspace root [tool.uv.sources] override for livekit
        cd "{{ justfile_directory() }}"
        if grep -q "^livekit = { path = " pyproject.toml; then
            echo "   Removing workspace root livekit source override..."
            sed -i.bak '/^livekit = { path = /d' pyproject.toml
            rm -f pyproject.toml.bak
        fi

        cd "{{ justfile_directory() }}/{{ agents_project }}"

        # Remove the [tool.uv.sources] override for livekit in agents project (if it exists)
        if grep -q "\[tool.uv.sources\]" pyproject.toml; then
            if grep -q "^livekit = " pyproject.toml; then
                echo "   Removing [tool.uv.sources] override..."
                sed -i.bak '/^livekit = /d' pyproject.toml
                rm -f pyproject.toml.bak
            fi
        fi

        # Remove any editable installation
        uv remove {{ package_name }} 2>/dev/null || true

        # Add the specific version from PyPI
        uv add "{{ package_name }}==$VERSION"
        uv sync

        # Clear any rust-sdk env var
        rm -f .env

        echo "✅ Linked to livekit $VERSION"

    # Case 3: No version - link to local python-rtc with default FFI version
    else
        echo "🐍 Linking local python-rtc with default FFI version..."

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

        # Download pre-built FFI artifacts (no --version, uses default from branch)
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

        echo "✅ Linked to local python-rtc (with default FFI version)"
    fi

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

# Build livekit-ffi from rust-sdks submodule
build-rust:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "🦀 Building livekit-ffi..."
    cd "{{ rust_submodule }}" && cargo build --release -p livekit-ffi
    echo "   Built: {{ rust_submodule }}/target/release"

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

    # Store absolute path before changing directory
    RUST_SUBMODULE_DIR="$(cd "{{ justfile_directory() }}/{{ rust_submodule }}" 2>/dev/null && pwd || echo "")"

    cd "{{ justfile_directory() }}/{{ agents_project }}"

    # Check if livekit is installed and how
    SHOW_OUTPUT=$(uv pip show livekit 2>/dev/null || echo "")
    IS_LOCAL_EDITABLE=false

    if [ -z "$SHOW_OUTPUT" ]; then
        echo "   livekit: NOT INSTALLED"
    elif echo "$SHOW_OUTPUT" | grep -q "Editable project location:"; then
        IS_LOCAL_EDITABLE=true
        VERSION=$(echo "$SHOW_OUTPUT" | grep "^Version:" | awk '{print $2}')
        LOCATION=$(echo "$SHOW_OUTPUT" | grep "Editable project location:" | cut -d' ' -f4-)
        echo "   livekit: LOCAL (editable) v$VERSION"
        echo "   path:    $LOCATION"
    else
        VERSION=$(echo "$SHOW_OUTPUT" | grep "^Version:" | awk '{print $2}')
        echo "   livekit: PyPI (v$VERSION)"
    fi

    echo ""

    # Check active FFI version
    if [ -f .env ] && grep -q "LIVEKIT_LIB_PATH" .env; then
        # Using locally built FFI from rust submodule
        SUBMODULE_COMMIT=$([ -n "$RUST_SUBMODULE_DIR" ] && cd "$RUST_SUBMODULE_DIR" 2>/dev/null && git rev-parse --short HEAD || echo 'unknown')
        echo "   FFI: LOCAL BUILD (rust-sdks @ $SUBMODULE_COMMIT)"
    elif [ "$IS_LOCAL_EDITABLE" = "true" ]; then
        # Local editable - check for pre-built or missing FFI
        FFI_PATH="$(cd "{{ justfile_directory() }}/{{ python_rtc }}" 2>/dev/null && pwd || echo "")/livekit/rtc/resources"
        if [ -d "$FFI_PATH" ] && { [ -f "$FFI_PATH/liblivekit_ffi.dylib" ] || [ -f "$FFI_PATH/liblivekit_ffi.so" ] || [ -f "$FFI_PATH/livekit_ffi.dll" ]; }; then
            # Extract version from Cargo.toml
            CARGO_TOML_PATH="$(cd "{{ justfile_directory() }}/{{ rust_submodule }}" 2>/dev/null && pwd || echo "")/livekit-ffi/Cargo.toml"
            if [ -f "$CARGO_TOML_PATH" ]; then
                FFI_VERSION=$(grep '^version = ' "$CARGO_TOML_PATH" | head -1 | sed 's/.*"\(.*\)".*/\1/')
                echo "   FFI: PRE-BUILT ARTIFACTS (v$FFI_VERSION)"
            else
                echo "   FFI: PRE-BUILT ARTIFACTS"
            fi
        else
            echo "   FFI: NOT AVAILABLE"
        fi
    else
        # PyPI version - FFI is bundled
        echo "   FFI: BUNDLED (PyPI package)"
    fi

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

    # Clean rust build from submodule
    cd "{{ justfile_directory() }}"
    cd "{{ rust_submodule }}" && cargo clean

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
    echo "rust-submodule: {{ rust_submodule }}"
    echo "package-name:   {{ package_name }}"
    echo "platform:       $PLATFORM (detected: $DETECTED_OS)"
    echo "arch:           $ARCH (detected: $DETECTED_ARCH)"