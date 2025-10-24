#!/bin/bash

# Directory containing the images
id_of_the_run=$1
INPUT_DIR="data/megadepth_test_1500/Undistorted_SfM/$id_of_the_run/images"
OUTPUT_DIR="data/megadepth/${id_of_the_run}"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Loop over image files (jpg, jpeg, png)
for img in "$INPUT_DIR"/*.{jpg,jpeg,png}; do
    # Skip if no matching files
    [ -e "$img" ] || continue

    filename=$(basename "$img")
    output="$OUTPUT_DIR/$filename"

    # Get image dimensions
    dimensions=$(identify -format "%w %h" "$img")
    width=$(echo $dimensions | cut -d' ' -f1)
    height=$(echo $dimensions | cut -d' ' -f2)

    # Calculate crop size and offsets
    if [ "$width" -gt "$height" ]; then
        offset=$(( (width - height) / 2 ))
        crop="${height}x${height}+$offset+0"
    else
        offset=$(( (height - width) / 2 ))
        crop="${width}x${width}+0+$offset"
    fi

    # Crop and resize
    convert "$img" -crop "$crop" +repage -resize 1024x1024 "$output"
done
