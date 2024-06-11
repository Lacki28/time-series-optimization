
        # Calculate the line number to split at (halfway point)
        split_line=$((total_lines / 2 + 1))

        # Use awk to select the second half of the file and overwrite the original file
        awk "NR >= $split_line" "$file" > "$file.tmp" && mv "$file.tmp" "$file"
    fi
done
