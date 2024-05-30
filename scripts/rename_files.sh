#!/bin/bash

# Loop through all files in the current directory
for file in *; do
    # Check if the file name contains the string 's20Loss20Ud'
    if [[ $file == *s20Loss20Ud* ]]; then
        # Replace 's20Loss20Ud' with 's10Loss10Ud' in the file name
        new_file="${file//s20Loss20Ud/s10Loss10Ud}"
        # Rename the file
        mv "$file" "$new_file"
    fi
done
