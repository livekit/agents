ARG TARGETARCH
FROM quay.io/pypa/manylinux_2_28_x86_64@sha256:b74a5b6f1705ecff17be8bd792589bef9b3b039766d12205720ecd2a9e919192 AS manylinux-amd64
FROM quay.io/pypa/manylinux_2_28_aarch64@sha256:0333f44c4eb759ec8f47b27bc9dc27b1c929932a267cf13ce0cff1a4b758cccb AS manylinux-arm64
FROM ghcr.io/astral-sh/uv@sha256:af22a52f5eafef3d78e551599d405c02df61919d18028f0cea8eda93bc9986d0 AS uv
# TARGETARCH selects one of the two digest-pinned stages above.
# hadolint ignore=DL3006
FROM manylinux-${TARGETARCH} AS wheel-builder
COPY --from=uv /uv /usr/local/bin/uv
COPY . /project
WORKDIR /project
ENV SOURCE_DATE_EPOCH=1733688115 \
    CMAKE_BUILD_PARALLEL_LEVEL=1 \
    PYTHON_CPU_COUNT=1 \
    UV_CONCURRENT_BUILDS=1 \
    UV_PYTHON_INSTALL_DIR=/opt/python3152
CMD ["/bin/bash", "/project/build_wheel.sh"]
