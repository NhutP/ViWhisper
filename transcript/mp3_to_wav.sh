#!/bin/bash

# Function to convert mp3 to wav while maintaining directory structure
convert_mp3_to_wav() {
  local input_file="$1"
  local input_dir="$2"
  local output_dir="$3"
  
  # Compute the relative path of the input file with respect to the input directory
  local relative_path="${input_file#$input_dir/}"
  
  # Create the corresponding output directory
  local output_file_dir="$(dirname "$output_dir/$relative_path")"
  mkdir -p "$output_file_dir"
  
  # Compute the output file path by changing the extension to .wav
  local output_file="${output_file_dir}/$(basename "$relative_path" .mp3).wav"

  # Check if the output file already exists
  if [ -f "$output_file" ]; then
    echo "Skipping conversion for $input_file, WAV file already exists."
    return
  fi

  # Convert the MP3 file to WAV
  ffmpeg -i "$input_file" -vn -acodec pcm_s16le -ar 16000 "$output_file"
}

# Export the function so it can be used by find -exec
export -f convert_mp3_to_wav

# Check if the correct number of arguments are provided
if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <input_directory> <output_directory>"
  exit 1
fi

# Input and output directories
input_dir="$1"
output_dir="$2"

# Find all mp3 files in the input directory and convert them to wav in the output directory
find "$input_dir" -type f -name "*.mp3" -exec bash -c 'convert_mp3_to_wav "$0" "$1" "$2"' {} "$input_dir" "$output_dir" \;

echo "Conversion completed."
