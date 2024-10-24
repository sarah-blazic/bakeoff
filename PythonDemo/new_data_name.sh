#!/bin/bash

# Parent directory
data_dir="data_old"

# Iterate over each subfolder (net, neutral, rim)
for folder in "$data_dir"/*/; do
    # Initialize a counter for each folder
    count=1
    # Iterate over each .wav file in the subfolder
    for file in "$folder"*.wav; do
        # Check if the file exists to avoid errors with an empty folder
        if [[ -f "$file" ]]; then
            # Get the directory and extension (always .wav here)
            dir=$(dirname "$file")
            extension="${file##*.}"
            # Rename file with sequential number and append "_old"
            new_filename="${count}_old.${extension}"
            mv "$file" "$dir/$new_filename"
            # Increment counter
            count=$((count + 1))
        fi
    done
done

