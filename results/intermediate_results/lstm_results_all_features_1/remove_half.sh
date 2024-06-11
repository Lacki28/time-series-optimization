#!/bin/bash

for file in *; do
    # Check if the item is a file (not a directory)
    if [ -f "$file" ]; then
        # Get the total number of lines in the file
        total_lines=$(wc -l < "$file")

        # Calculate the line number to split at (halfway point)
        split_line=$((total_lines / 2))

        # Use awk to select the first half of the file and overwrite the original file
        awk "NR <= $split_line" "$file" > "$file.tmp" && mv "$file.tmp" "$file"
    fi
done
