#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake .. -DCMAKE_BUILD_TYPE=Release
make -j"$(nproc)"

# Copy the .so into the repo root so `import blade` works from there.
EXT=$(python3-config --extension-suffix)
cp blade"$EXT" "$SCRIPT_DIR/"
echo "Built: $SCRIPT_DIR/blade$EXT"
