#!/usr/bin/env bash
set -e

ROOT_DIR="Long3D"

for scene_dir in "$ROOT_DIR"/*; do
    if [ -d "$scene_dir" ]; then
        archive="$scene_dir/images.7z"
        if [ -f "$archive" ]; then
            echo "==> Extracting $archive"
            7z x "$archive" -o"$scene_dir"
        else
            echo "!! Skip $scene_dir (no images.7z)"
        fi
    fi
done

echo "All done."
