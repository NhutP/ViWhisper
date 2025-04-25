import os
import csv
from collections import defaultdict

def merge_csv_files_by_channel(directory, output_folder):
    # Dictionary to store file paths for each channel
    channel_files = defaultdict(list)
    
    # List all files in the directory
    all_files = [f for f in os.listdir(directory) if f.startswith('video_info_') and f.endswith('.csv')]
    
    # Group files by channel name
    for filename in all_files:
        # Extract the channel name from the filename
        parts = filename.split('_')
        if len(parts) >= 3:
            channel_name = parts[2]
            channel_files[channel_name].append(filename)
    
    # Create output folder if it does not exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Merge files for each channel
    for channel_name, files in channel_files.items():
        output_file = os.path.join(output_folder, f'merged_video_info_{channel_name}.csv')
        seen_headers = set()
        
        with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.writer(outfile)
            for i, filename in enumerate(files):
                file_path = os.path.join(directory, filename)
                with open(file_path, 'r', encoding='utf-8') as infile:
                    reader = csv.reader(infile)
                    headers = next(reader)
                    
                    # Write the header only for the first file
                    if i == 0:
                        writer.writerow(headers)
                        seen_headers.add(tuple(headers))
                    elif tuple(headers) not in seen_headers:
                        writer.writerow(headers)
                        seen_headers.add(tuple(headers))
                    
                    for row in reader:
                        writer.writerow(row)
                    
        print(f"Đã gộp các file CSV vào {output_file} cho kênh {channel_name}")

if __name__ == "__main__":
    directory = '.'  # Thư mục chứa các file video_info
    output_folder = 'merged_files'  # Thư mục để lưu các file merged
    merge_csv_files_by_channel(directory, output_folder)
