import argparse
import requests
import csv
from isodate import parse_duration
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from youtube_transcript_api.formatters import TextFormatter
import os
from pytube import YouTube, exceptions
from urllib.parse import urlparse, parse_qs
import re
from googleapiclient.discovery import build
from datetime import datetime

def get_channel_id_from_custom_name(custom_name, api_key):
    youtube = build('youtube', 'v3', developerKey=api_key)
    request = youtube.search().list(
        part='id',
        q=custom_name,
        type='channel'
    )
    response = request.execute()
    if response['items']:
        return response['items'][0]['id']['channelId']
    return None

def get_channel_id(youtube_url, api_key):
    handle_regex = r"(?:youtube\.com\/@)([a-zA-Z0-9\-_]+)"
    match = re.search(handle_regex, youtube_url)
    if match:
        handle_name = match.group(1)
        return get_channel_id_from_custom_name(handle_name, api_key)
    return None

def fetch_and_save_video_details(api_key, channel_id, channel_name, duration_limit, csv_file_path, max_results=50, total_limit=170):
    url = f"https://www.googleapis.com/youtube/v3/search?key={api_key}&channelId={channel_id}&part=snippet,id&order=date&maxResults={max_results}"
    video_count = 0
    total_duration = 0
    file_exists = os.path.isfile(csv_file_path)

    with open(csv_file_path, 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Tên video", "Độ dài video (giây)", "Link video", "Bản chép lời tiếng Việt"])

        while url and video_count < total_limit:
            try:
                response = requests.get(url)
                data = response.json()
            except Exception as e:
                print(f"Lỗi khi gọi API: {e}")
                break

            if 'error' in data:
                raise Exception(data['error']['message'])

            videos = data.get('items', [])

            for video in videos:
                if video_count >= total_limit:
                    break
                if video['id']['kind'] == "youtube#video":
                    video_count += 1
                    print(f"Đang xử lý video số: {video_count}")

                    video_id = video['id']['videoId']
                    video_title = video['snippet']['title']
                    video_url = f"https://www.youtube.com/watch?v={video_id}"

                    video_details_url = f"https://www.googleapis.com/youtube/v3/videos?id={video_id}&key={api_key}&part=contentDetails"
                    response = requests.get(video_details_url)
                    details = response.json().get('items', [])
                    if not details:
                        continue
                    video_duration = parse_duration(details[0]['contentDetails']['duration']).total_seconds()
                    total_duration += video_duration

                    try:
                        yt = YouTube(video_url)
                        audio_stream = yt.streams.filter(only_audio=True).first()
                    except exceptions.AgeRestrictedError:
                        print(f"Không thể tải xuống video từ URL {video_url} do giới hạn độ tuổi.")
                        total_duration -= video_duration
                        continue
                    except Exception as e:
                        print(f"Không thể tải audio cho video {video_url}: {e}")
                        total_duration -= video_duration
                        continue

                    transcript_available = check_transcript_availability(video_id)
                    if transcript_available:
                        writer.writerow([video_title, video_duration, video_url, transcript_available])
                    else:
                        total_duration -= video_duration
                        continue

                    if total_duration > duration_limit or video_count >= total_limit:
                        print(f"Đã đạt giới hạn thời gian hoặc số lượng video cho kênh {channel_name}. Dừng xử lý thêm video.")
                        return total_duration, True

            url = data.get('nextPageToken')
            if url:
                url = f"https://www.googleapis.com/youtube/v3/search?key={api_key}&channelId={channel_id}&part=snippet,id&order=date&maxResults={max_results}&pageToken={url}"
            else:
                break

    if video_count < total_limit:
        print(f"Kênh {channel_name} không có đủ số lượng video hoặc thời lượng yêu cầu. Đánh dấu hoàn thành.")
        return total_duration, True

    print(f"Tổng số video đã xử lý: {video_count}")
    return total_duration, video_count >= 100

def check_transcript_availability(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['vi'])
        if transcript:
            return f"https://www.youtube.com/api/timedtext?v={video_id}&lang=vi&fmt=vtt&name="
    except (TranscriptsDisabled, NoTranscriptFound):
        print(f"Không có bản chép lời cho video ID {video_id}")
    except Exception as e:
        print(f"Không thể kiểm tra bản chép lời cho video ID {video_id}: {e}")
    return None

def download_youtube_transcript(youtube_url, index, channel_name, flag, output_folder):
    url_data = urlparse(youtube_url)
    query_params = parse_qs(url_data.query)
    video_id = query_params.get('v', [None])[0]

    if not video_id:
        print(f"Không thể lấy video ID từ URL: {youtube_url}")
        return

    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['vi'])
        formatter = TextFormatter()
        transcript_text = formatter.format_transcript(transcript)

        channel_folder = os.path.join(output_folder, f"{channel_name}_{flag}")
        os.makedirs(channel_folder, exist_ok=True)
        filename = os.path.join(channel_folder, f"{index:09d}.txt")
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(transcript_text)
        print(f"Bản chép lời đã được lưu: {filename}")
    except (TranscriptsDisabled, NoTranscriptFound):
        print(f"Không có bản chép lời cho video {youtube_url}")
    except Exception as e:
        print(f"Không thể lấy bản chép lời cho video {youtube_url}: {e}")

def download_transcripts_from_csv(csv_file_path, channel_name, flag, output_folder):
    with open(csv_file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        index = 1
        for row in reader:
            if len(row) > 2:
                youtube_url = row[2]
                download_youtube_transcript(youtube_url, index, channel_name, flag, output_folder)
                index += 1

def download_youtube_audio(youtube_url, index, channel_name, flag, output_folder):
    try:
        yt = YouTube(youtube_url)
        audio_stream = yt.streams.filter(only_audio=True).first()
        channel_folder = os.path.join(output_folder, f"{channel_name}_{flag}")
        os.makedirs(channel_folder, exist_ok=True)
        out_file = audio_stream.download(output_path=channel_folder)
        base, ext = os.path.splitext(out_file)
        new_file = os.path.join(channel_folder, f'{index:09d}.mp3')
        os.rename(out_file, new_file)
        print(f"Audio đã được lưu: {new_file}")
    except exceptions.AgeRestrictedError:
        print(f"Không thể tải xuống video từ URL {youtube_url} do giới hạn độ tuổi: {youtube_url}")
    except Exception as e:
        print(f"Không thể tải audio cho video {youtube_url}: {e}")

def download_audios_from_csv(csv_file_path, channel_name, flag, output_folder):
    with open(csv_file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        index = 1
        for row in reader:
            if len(row) > 2:
                youtube_url = row[2]
                download_youtube_audio(youtube_url, index, channel_name, flag, output_folder)
                index += 1

def process_youtube_channel(api_key, channel_link, channel_name, flag, duration_limit, transcript_folder, audio_folder, video_info_folder, max_results):
    channel_id = get_channel_id(channel_link, api_key)
    if not channel_id:
        print(f"Không thể lấy ID kênh từ link {channel_link}")
        return 0, False

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_file_path = os.path.join(video_info_folder, f'{channel_name}_{flag}', f'video_info_{channel_name}_{flag}_{timestamp}.csv')

    os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)

    total_duration, is_complete = fetch_and_save_video_details(api_key, channel_id, channel_name, duration_limit, csv_file_path, max_results)
    if is_complete:
        download_transcripts_from_csv(csv_file_path, channel_name, flag, transcript_folder)
        download_audios_from_csv(csv_file_path, channel_name, flag, audio_folder)
    return total_duration, is_complete

def update_channel_completion(channel_link, channels_file, channel_index):
    updated_rows = []
    with open(channels_file, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        headers = next(reader)
        for index, row in enumerate(reader):
            if index == channel_index:
                row[3] = '1'
            updated_rows.append(row)

    with open(channels_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(updated_rows)

def main(api_keys_file, channels_file, transcript_folder, audio_folder, video_info_folder, max_results):
    with open(api_keys_file, 'r') as f:
        api_keys = f.read().splitlines()

    with open(channels_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        channels = list(reader)

    current_key_index = 0

    for channel_index, channel in enumerate(channels):
        channel_link, duration_limit, flag, is_complete = channel
        flag = int(flag)
        is_complete = int(is_complete)
        duration_limit = int(duration_limit)

        if is_complete == 1:
            continue

        while current_key_index < len(api_keys):
            api_key = api_keys[current_key_index]
            try:
                print(f"Đang xử lý kênh {channel_link} với API key {api_key}")
                channel_name = os.path.basename(os.path.normpath(channel_link.split('/')[-1]))
                total_duration, is_complete = process_youtube_channel(api_key, channel_link, channel_name, flag, duration_limit, transcript_folder, audio_folder, video_info_folder, max_results)
                if is_complete:
                    update_channel_completion(channel_link, channels_file, channel_index)
                    print(f"Tổng thời lượng video cho kênh {channel_link}: {total_duration} giây")
                    break
                elif flag == 0:
                    break
                else:
                    flag = 0
                    print("Đã set flag = 0")

                if current_key_index >= len(api_keys):
                    with open('uncomplete.txt', 'a') as incomplete_file:
                        incomplete_file.write(f"{channel_link},{duration_limit}\n")
                    print("Duyệt hết channel. Lưu trạng thái và thoát.")
                    return

            except Exception as e:
                print(f"API key {api_key} gặp lỗi: {e}")
                current_key_index += 1
                if flag == 0:
                    break
                else:
                    flag = 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process YouTube channels and download transcripts and audio.')
    parser.add_argument('api_keys_file', type=str, help='Path to the API keys file.')
    parser.add_argument('channels_file', type=str, help='Path to the channels file.')
    parser.add_argument('transcript_folder', type=str, help='Path to the folder where transcripts will be saved.')
    parser.add_argument('audio_folder', type=str, help='Path to the folder where audio files will be saved.')
    parser.add_argument('video_info_folder', type=str, help='Path to the folder where video info files will be saved.')
    parser.add_argument('--max_results', type=int, default=50, help='Maximum number of results to retrieve per channel.')
    args = parser.parse_args()

    main(args.api_keys_file, args.channels_file, args.transcript_folder, args.audio_folder, args.video_info_folder, args.max_results)
