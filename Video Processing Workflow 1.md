# File: export.py

## Summary

This Python module provides functionality to export processed video data to a JSON file. It is designed to handle video metadata, tags, and classification, and save this information in a specified directory. This module is a part of a larger video processing application, where it serves the purpose of persisting processed data for further analysis or sharing.

## Changes Made

- Added function 'export_to_json' to handle exporting video data to JSON files.

```python
import json
import logging
import os
from core import ensure_directory


def export_to_json(data, output_dir):
    """Export processed video data to a JSON file.

Args:
    data (Any): Processed video data including metadata, tags, and classification.
    output_dir (Any): Directory where the JSON file should be exported.

Returns:
    Any: The result of the export operation, which may include success status or error messages."""
    ensure_directory(output_dir)
    filename = os.path.join(output_dir, f"{data['filename']}.json")
    try:
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        logging.info(f'Data exported to {filename}')
    except Exception as e:
        logging.error(f'Failed to export data to {filename}: {e}')

```

# File: qwen_model.py

## Summary

This Python module defines the QwenVLModel class, which is designed for processing video data using the Qwen2-VL model. The class provides methods to initialize the model and process video frames with associated metadata and instructions. It is a part of a larger application that handles video analysis and processing tasks.

## Changes Made

- Added QwenVLModel class to handle video processing.
- Implemented __init__ method for initializing the model with a specified directory.
- Implemented process_video method for processing video frames with metadata and instructions.

```python
import os
import logging
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from modelscope import snapshot_download
from qwen_vl_utils import process_vision_info
import torch


class QwenVLModel:
    """Represents the Qwen2-VL model, which is designed for processing video data with associated metadata and instructions."""

    def __init__(self, model_dir='/path/to/your/qwen2-vl-model'):
        """Initializes the Qwen2-VL model and processor.

    Args:
        self (Any): The instance of the class.
        model_dir (Any): The directory where the model is stored or the model configuration.

    Returns:
        Any: An instance of the QwenVLModel class, initialized with the specified model directory."""
        try:
            self.model_path = snapshot_download('qwen/Qwen2-VL-7B-Instruct',
                cache_dir=model_dir)
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(self
                .model_path, torch_dtype='auto', device_map='auto')
            self.model.eval()
            self.processor = AutoProcessor.from_pretrained(self.model_path)
            logging.info('Qwen2-VL model loaded successfully.')
        except Exception as e:
            logging.error(f'Error loading Qwen2-VL model: {e}')
            raise

    def process_video(self, video_frames, metadata, instruction=
        'Describe this video.'):
        """Processes a video using the Qwen2-VL model.

    Args:
        self (Any): The instance of the class.
        video_frames (Any): The frames of the video to be processed, typically in a format suitable for the model.
        metadata (Any): Additional information related to the video, which may influence processing.
        instruction (Any): Specific instructions or parameters that guide the processing of the video.

    Returns:
        Any: The result of processing the video, which may include predictions, annotations, or processed frames."""
        messages = [{'role': 'user', 'content': [{'type': 'video', 'video':
            video_frames, 'fps': metadata.get('fps', 1)}, {'type': 'text',
            'text': instruction}]}]
        try:
            text = self.processor.apply_chat_template(messages, tokenize=
                False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(text=[text], images=image_inputs,
                videos=video_inputs, padding=True, return_tensors='pt')
            inputs = inputs.to('cuda' if torch.cuda.is_available() else 'cpu')
            generated_ids = self.model.generate(**inputs, max_new_tokens=128)
            generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids,
                out_ids in zip(inputs.input_ids, generated_ids)]
            output_text = self.processor.batch_decode(generated_ids_trimmed,
                skip_special_tokens=True, clean_up_tokenization_spaces=False)
            return output_text[0]
        except Exception as e:
            logging.error(f'Error processing video with Qwen2-VL: {e}',
                exc_info=True)
            return None

```

# File: interface.py

## Summary

This Python script is designed to process video data by converting CSV metadata to JSON, running inference on video segments, and managing command-line arguments. It serves as a utility for handling video processing tasks, including format conversion and metadata extraction, within a larger video analysis application.

## Changes Made

- Added function 'csv_to_json' to convert CSV files to JSON format with resolution and FPS constraints.
- Implemented 'run_inference_on_videos' to perform inference on video segments.
- Created 'parse_arguments' to handle command-line argument parsing.
- Developed 'read_video_metadata_from_csv' to extract video metadata from CSV files.
- Introduced 'main' function to orchestrate the video processing workflow.

```python
import argparse
import logging
from core import load_config, ensure_directory, setup_logging, query_qwen_vl_chat
from analysis import segment_video
from workflow import process_video_pipeline
import json
import csv
import os


def csv_to_json(csv_file, output_json_file, max_resolution='720p', max_fps=30):
    """Converts CSV to JSON with resolution and FPS limits.

Args:
    csv_file (Any): The input CSV file to be converted.
    output_json_file (Any): The output JSON file where the converted data will be saved.
    max_resolution (Any): The maximum resolution allowed for the video data.
    max_fps (Any): The maximum frames per second allowed for the video data.

Returns:
    Any: The result of the conversion process, which may include success status or error messages."""
    videos = []
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            video_data = {'file_name': row['file_name'], 'file_size': row[
                'file_size'], 'last_modified': row['last_modified'],
                'public_url': row['public_url'], 'segments': [],
                'max_resolution': max_resolution, 'max_fps': max_fps}
            videos.append(video_data)
    for video in videos:
        output_dir = os.path.join('segmented_videos', video['file_name'])
        segment_video(video['public_url'], output_dir, 15, video[
            'max_resolution'], video['max_fps'])
        for i in range(1, 999):
            segment_url = f"{video['public_url'][:-4]}_clip_{i:03d}.mp4"
            if os.path.exists(os.path.join(output_dir, f'clip_{i:03d}.mp4')):
                video['segments'].append(segment_url)
            else:
                break
    with open(output_json_file, 'w') as json_file:
        json.dump(videos, json_file, indent=2)
        print(f'JSON output successfully written to {output_json_file}')


def run_inference_on_videos(videos):
    """Runs inference on video segments, handling resolution and FPS.

Args:
    videos (Any): A collection of video segments on which inference will be performed.

Returns:
    Any: The results of the inference process, which may include predictions or analysis results."""
    all_results = []
    for video in videos:
        file_name = video['file_name']
        segment_results = []
        for segment_url in video['segments']:
            description = query_qwen_vl_chat(segment_url, instruction=
                'Describe this video segment.', max_resolution=video[
                'max_resolution'], max_fps=video['max_fps'])
            segment_results.append({'segment_url': segment_url,
                'description': description})
        all_results.append({'file_name': file_name, 'segments':
            segment_results})
    return all_results


def parse_arguments():
    """Parses command-line arguments provided by the user.

Returns:
    Namespace: Parsed command-line arguments, which can be accessed as attributes."""
    config = load_config()
    parser = argparse.ArgumentParser(description=
        'Comprehensive Video Processing Toolkit')
    default_output = config.get('Paths', 'ProcessedDirectory', fallback=
        'processed_videos/')
    default_scene_threshold = config.getfloat('SceneDetection',
        'DefaultThreshold', fallback=0.3)
    default_qwen_instruction = config.get('QwenVL', 'DefaultInstruction',
        fallback='Describe this video.')
    parser.add_argument('--urls', nargs='+', help=
        'List of video URLs to process')
    parser.add_argument('--csv', type=str, help=
        'Path to the CSV file containing video URLs and metadata')
    parser.add_argument('--config', type=str, help=
        'Path to the configuration file', default='config.ini')
    parser.add_argument('--output', type=str, help='Output directory',
        default=default_output)
    parser.add_argument('--log_level', type=str, help=
        'Set the logging level', choices=['DEBUG', 'INFO', 'WARNING',
        'ERROR', 'CRITICAL'], default='INFO')
    parser.add_argument('--scene_threshold', type=float, help=
        'Threshold for scene detection', default=default_scene_threshold)
    parser.add_argument('--qwen_instruction', type=str, help=
        'Instruction passed to Qwen-VL', default=default_qwen_instruction)
    parser.add_argument('--use_vpc', action='store_true', help=
        'Use the VPC endpoint for Qwen-VL API calls')
    args = parser.parse_args()
    if not args.urls and not args.csv:
        parser.error(
            'You must provide either --urls or --csv to indicate which videos to process.'
            )
    return args


def read_video_metadata_from_csv(csv_file):
    """Reads video metadata from a CSV file created with rclone.

Args:
    csv_file (str): Path to the CSV file containing video metadata.

Returns:
    list: A list of dictionaries containing video metadata (e.g., filename, file_size, public_url)."""
    videos = []
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        required_columns = ['file_name', 'file_size', 'last_modified',
            'public_url']
        for col in required_columns:
            if col not in reader.fieldnames:
                logging.error(
                    f"The CSV file must contain the following columns: {', '.join(required_columns)}"
                    )
                return []
        for row in reader:
            if row.get('public_url'):
                videos.append({'file_name': row.get('file_name'),
                    'file_size': row.get('file_size'), 'last_modified': row
                    .get('last_modified'), 'public_url': row.get('public_url')}
                    )
    return videos


def main():
    """Main function to handle video processing.

This function orchestrates the overall video processing workflow, including reading metadata, running inference, and converting formats.

Returns:
    Any: The outcome of the video processing, which may include success status or error messages."""
    args = parse_arguments()
    setup_logging(log_file='video_processing.log')
    logging.getLogger().setLevel(args.log_level)
    config = load_config(args.config)
    ensure_directory(args.output)
    if args.csv:
        logging.info(f'Reading videos from CSV file: {args.csv}')
        max_resolution = args.max_resolution or config.get('Processing',
            'MaxResolution', fallback='720p')
        max_fps = args.max_fps or config.getint('Processing', 'MaxFPS',
            fallback=30)
        csv_to_json(args.csv, 'videos_with_segments.json', max_resolution,
            max_fps)
        with open('videos_with_segments.json', 'r') as f:
            videos = json.load(f)
        inference_results = run_inference_on_videos(videos)
    else:
        videos = [{'public_url': url} for url in args.urls]
        for video_info in videos:
            video_url = video_info.get('public_url')
            filename = video_info.get('file_name', None)
            logging.info(f'Processing video: {filename} from URL: {video_url}')
            process_video_pipeline(video_url=video_url, scene_threshold=
                args.scene_threshold, custom_rules=None, output_dir=args.output
                )


if __name__ == '__main__':
    main()

```

# File: analysis.py

## Summary

This Python module provides a set of functions for video processing tasks, including segmenting videos, extracting metadata, detecting scene changes, and extracting specific scene segments. It leverages libraries such as FFmpeg and MoviePy to perform these operations efficiently. The module is designed to facilitate video editing and analysis by providing tools to manipulate video files programmatically.

## Changes Made

- Added function 'segment_video' to segment videos into smaller clips with specified resolution and FPS limits.
- Added function 'extract_metadata' to extract video metadata using FFmpeg.
- Added function 'detect_scenes' to detect scene changes in a video using the 'scenedetect' library.
- Added function 'extract_scene_segment' to extract specific scene segments from a video using MoviePy.

```python
import subprocess
import logging
import csv
import json
import os
from core import query_qwen_vl_chat, ensure_directory
from moviepy.editor import VideoFileClip
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector


def segment_video(input_video, output_dir, segment_length=15,
    max_resolution='720p', max_fps=30):
    """Segments a video into smaller clips with specified resolution and FPS limits.

Args:
    input_video (str): Path to the input video file.
    output_dir (str): Directory to save the segmented clips.
    segment_length (int, optional): Length of each segment in seconds. Defaults to 15.
    max_resolution (str, optional): Maximum resolution for the output clips (e.g., "720p", "1080p"). Defaults to "720p".
    max_fps (int, optional): Maximum frame rate for the output clips. Defaults to 30.

Returns:
    Any: The result of the segmentation process, which may include information about the created segments or an error message if the process fails."""
    try:
        ensure_directory(output_dir)
        if max_resolution == '720p':
            scale_filter = (
                'scale=trunc(min(iw\\,1280)/2)*2:trunc(min(ih\\,720)/2)*2')
        elif max_resolution == '1080p':
            scale_filter = (
                'scale=trunc(min(iw\\,1920)/2)*2:trunc(min(ih\\,1080)/2)*2')
        else:
            scale_filter = (
                'scale=trunc(min(iw\\,1280)/2)*2:trunc(min(ih\\,720)/2)*2')
        command = ['ffmpeg', '-i', input_video, '-vf',
            f'{scale_filter},fps={max_fps}', '-c:v', 'libx264', '-c:a',
            'copy', '-f', 'segment', '-segment_time', str(segment_length),
            f'{output_dir}/clip_%03d.mp4']
        subprocess.run(command, check=True)
        logging.info(
            f"Video '{input_video}' segmented into {segment_length}-second clips."
            )
    except subprocess.CalledProcessError as e:
        logging.error(f'Error segmenting video: {e}')


def extract_metadata(filepath):
    """Extracts video metadata using FFmpeg.

Args:
    filepath (str): The path to the video file.

Returns:
    dict: A dictionary containing the extracted metadata, or None if extraction fails."""
    command = ['ffmpeg', '-i', filepath, '-v', 'quiet', '-print_format',
        'json', '-show_format', '-show_streams']
    try:
        output = subprocess.check_output(command)
        metadata = json.loads(output.decode('utf-8'))
        video_stream = None
        for stream in metadata['streams']:
            if stream['codec_type'] == 'video':
                video_stream = stream
                break
        if video_stream:
            logging.info(f'Extracted metadata for {filepath}')
            return {'duration': float(metadata['format']['duration']),
                'bit_rate': int(metadata['format']['bit_rate']),
                'resolution':
                f"{video_stream['width']}x{video_stream['height']}",
                'frame_rate': video_stream['r_frame_rate'], 'codec':
                video_stream['codec_name'], 'size': os.path.getsize(filepath)}
        else:
            logging.warning(f'No video stream found in {filepath}')
            return None
    except subprocess.CalledProcessError as e:
        logging.error(f'Failed to extract metadata: {e}')
        return None


def detect_scenes(filepath, threshold=30.0):
    """Detects scene changes in a video using the `scenedetect` library.

Args:
    filepath (str): The path to the video file.
    threshold (float, optional): Sensitivity threshold for scene detection. Defaults to 30.0.

Returns:
    list: A list of dictionaries, each containing the start and end time of a scene, or an empty list if no scenes are detected."""
    video_manager = VideoManager([filepath])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))
    video_manager.set_downscale_factor()
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)
    scenes = scene_manager.get_scene_list()
    scene_list = [{'start_time': scene[0].get_seconds(), 'end_time': scene[
        1].get_seconds()} for scene in scenes]
    logging.info(f'Detected {len(scene_list)} scenes in {filepath}')
    return scene_list


def extract_scene_segment(video_path, start_time, end_time, output_dir=
    'scene_segments'):
    """Extracts a scene segment from a video using MoviePy.

Args:
    video_path (str): The path to the video file from which the segment will be extracted.
    start_time (float): The start time of the segment in seconds.
    end_time (float): The end time of the segment in seconds.
    output_dir (str): Directory where the extracted segment will be saved.

Returns:
    Any: The result of the extraction process, which may include information about the created segment or an error message if the process fails."""
    ensure_directory(output_dir)
    filename = os.path.basename(video_path).split('.')[0]
    segment_path = os.path.join(output_dir,
        f'{filename}_{start_time:.2f}-{end_time:.2f}.mp4')
    try:
        video = VideoFileClip(video_path)
        segment = video.subclip(start_time, end_time)
        segment.write_videofile(segment_path, codec='libx264', audio_codec=
            'aac')
        video.close()
        logging.info(f'Extracted scene segment: {segment_path}')
        return segment_path
    except Exception as e:
        logging.error(f'Error extracting scene segment: {e}')
        return None

```

# File: core.py

## Summary

This Python module provides utility functions for configuration management, logging setup, directory management, database connection, keyword loading, and AI model interaction. It is designed to support a larger application by handling essential setup and operational tasks, such as loading configurations, ensuring necessary directories exist, connecting to databases, and interacting with AI models for video processing.

## Changes Made

- Added function `load_config` to load configuration files.
- Added function `setup_logging` to configure logging settings.
- Added function `ensure_directory` to verify or create directories.
- Added function `connect_to_mongodb` to establish a database connection.
- Added function `get_mongodb_collection` to retrieve a MongoDB collection.
- Added function `load_priority_keywords` to load keywords from a file.
- Added function `query_qwen_vl_chat` to interact with the Qwen-VL-Chat model API.

```python
import requests
import configparser
import logging
import os
import json
from logging.handlers import RotatingFileHandler
config = configparser.ConfigParser()


def load_config(config_path='config.ini'):
    """Loads the configuration from a config file.

Args:
    config_path (Any): The path to the configuration file, which can be a string or any other type that represents a file path.

Returns:
    Any: The loaded configuration data, which can be of any type depending on the configuration file format."""
    try:
        config.read(config_path)
        return config
    except configparser.Error as e:
        logging.error(f'Error reading config.ini: {e}')
        raise


def setup_logging(log_file='video_processing.log'):
    """Set up logging configuration.

Args:
    log_file (Any): The path to the log file where logs will be written. This can be a string or any other type that represents a file path.

Returns:
    Any: None, but configures the logging settings for the application."""
    log_handler = RotatingFileHandler(log_file, maxBytes=1024 * 1024,
        backupCount=5)
    logging.basicConfig(level=logging.INFO, format=
        '%(asctime)s - %(levelname)s - %(message)s', handlers=[log_handler])


def ensure_directory(directory):
    """Ensures that a directory exists.

Args:
    directory (Any): The path to the directory that needs to be checked or created. This can be a string or any other type that represents a directory path.

Returns:
    Any: None, but creates the directory if it does not exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        logging.info(f'Created directory: {directory}')
    else:
        logging.debug(f'Directory exists: {directory}')


def connect_to_mongodb():
    """Establishes a connection to MongoDB/CosmosDB.

Returns:
    Any: A connection object or client instance that can be used to interact with the database."""
    try:
        from pymongo import MongoClient
        db_uri = config.get('MongoDB', 'URI', fallback='your-default-uri')
        client = MongoClient(db_uri)
        client.admin.command('ping')
        return client
    except Exception as e:
        logging.error(f'MongoDB connection failed: {e}')
        return None


def get_mongodb_collection():
    """Retrieves the MongoDB collection after establishing a connection.

Returns:
    Any: The MongoDB collection object that can be used to perform database operations."""
    try:
        client = connect_to_mongodb()
        database_name = config.get('MongoDB', 'DatabaseName', fallback=
            'video_database')
        collection_name = config.get('MongoDB', 'CollectionName', fallback=
            'videos')
        db = client[database_name]
        collection = db[collection_name]
        return collection
    except Exception as e:
        logging.error(f'Could not retrieve MongoDB collection: {e}')
        return None


def load_priority_keywords(file_path='priority_keywords.json'):
    """Load priority keywords for analysis.

Args:
    file_path (Any): The path to the file containing priority keywords. This can be a string or any other type that represents a file path.

Returns:
    Any: A list or set of priority keywords loaded from the specified file."""
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as file:
                return json.load(file)
        except Exception as e:
            logging.error(f'Could not load priority keywords: {e}')
    return []


def query_qwen_vl_chat(video_url, instruction='Describe this video.',
    use_vpc=False):
    """Sends a request to the Qwen-VL-Chat model API over HTTPS to process a video.

Args:
    video_url (str): The URL of the video file.
    instruction (str): The instruction or query to be passed to the AI model.
    use_vpc (bool): Whether to use the VPC endpoint instead of the public one.

Returns:
    str: The model's response, typically a description or analysis of the video."""
    public_endpoint = config.get('QwenVL', 'PublicEndpoint', fallback=
        'https://your-public-endpoint')
    vpc_endpoint = config.get('QwenVL', 'VpcEndpoint', fallback=
        'https://your-vpc-endpoint')
    access_key = config.get('QwenVL', 'AccessKey', fallback='your-access-key')
    endpoint = vpc_endpoint if use_vpc else public_endpoint
    priority_keywords = load_priority_keywords()
    if priority_keywords:
        instruction += ' Focus on these aspects: ' + ', '.join(
            priority_keywords)
    payload = {'video_url': video_url, 'instruction': instruction}
    headers = {'Authorization': f'Bearer {access_key}', 'Content-Type':
        'application/json'}
    try:
        response = requests.post(endpoint, json=payload, headers=headers)
        if response.status_code == 200:
            result = response.json()
            return result.get('description', 'No description available.')
        else:
            logging.error(f'API Error {response.status_code}: {response.text}')
            return None
    except requests.exceptions.RequestException as e:
        logging.error(f'Failed to connect to the API: {e}')
        return None

```

# File: tagging.py

## Summary

This Python module is designed to handle video tagging and classification based on metadata, AI-generated descriptions, and custom rules. It provides functions to load custom tagging rules, apply tags to videos, apply custom rules, and classify videos into high-level categories. This module is a part of a larger application that manages video content by organizing and categorizing it for better accessibility and searchability.

## Changes Made

- Added function 'load_custom_tags' to load custom tagging rules from a JSON file.
- Added function 'apply_tags' to apply tags to videos based on metadata and custom rules.
- Added function 'apply_custom_rules' to modify or add tags using custom rules.
- Added function 'classify_video' to classify videos into high-level categories based on tags and metadata.

```python
import logging
import json
import os


def load_custom_tags(custom_rules_path):
    """Loads custom tagging rules from a JSON file.

Args:
    custom_rules_path (Any): The path to the JSON file containing custom tagging rules.

Returns:
    Any: The content of the JSON file, which may include custom tagging rules or an error message if loading fails."""
    if not custom_rules_path or not os.path.isfile(custom_rules_path):
        logging.warning(f'Custom tag rules file {custom_rules_path} not found.'
            )
        return {}
    with open(custom_rules_path, 'r') as file:
        return json.load(file)


def apply_tags(video_info, custom_rules=None):
    """Applies tags to a video based on its metadata, AI-generated description, and custom rules.

Args:
    video_info (dict): A dictionary containing video metadata and AI-generated description.
    custom_rules (str or None): Path to the custom rules JSON file.

Returns:
    list: List of applied tags, which may include default tags and any additional tags defined in custom rules."""
    tags = []
    if 'resolution' in video_info and '4K' in video_info['resolution']:
        tags.append('High Resolution')
    if 'duration' in video_info and video_info['duration'] > 600:
        tags.append('Extended Play')
    if video_info.get('codec') == 'h264':
        tags.append('H.264 Codec')
    description = video_info.get('qwen_description', '').lower()
    if 'action' in description:
        tags.append('Action')
    if 'night shot' in description:
        tags.append('Night-time Filming')
    if custom_rules:
        custom_tags_rules = load_custom_tags(custom_rules)
        tags = apply_custom_rules(video_info, tags, custom_tags_rules)
    return list(set(tags))


def apply_custom_rules(video_info, tags, custom_tags_rules):
    """Applies custom rules provided in a JSON file to tag the video.

Args:
    video_info (dict): A dictionary containing video metadata and AI-generated descriptions.
    tags (list): List of already applied tags during the default rules processing.
    custom_tags_rules (dict): Dictionary of custom rules that specify how to modify or add tags.

Returns:
    list: Updated list of tags after applying custom rules, which may include modifications to existing tags or new tags added based on the custom rules."""
    rules = custom_tags_rules.get('rules', [])
    for rule in rules:
        if 'description_keywords' in rule:
            keywords = set(rule.get('description_keywords', []))
            if keywords.intersection(set(video_info.get('qwen_description',
                '').split())):
                tags.extend(rule.get('tags', []))
        if 'metadata_conditions' in rule:
            conditions = rule.get('metadata_conditions', {})
            resolution = video_info.get('resolution', '')
            duration = video_info.get('duration', 0)
            if conditions.get('resolution'
                ) == resolution and duration > conditions.get('duration_gt', 0
                ):
                tags.extend(rule.get('tags', []))
    return list(set(tags))


def classify_video(video_info):
    """Classifies a video into a high-level category based on its tags and metadata.

Args:
    video_info (dict): A dictionary containing tagged video metadata, which is used to determine the classification.

Returns:
    str: Classification of the video (e.g., "Action", "Documentary"), representing the high-level category the video belongs to."""
    tags = video_info.get('tags', [])
    if 'Action' in tags or 'High Intensity' in tags:
        return 'Action'
    if 'Documentary Footage' in tags or 'Narrative' in video_info.get(
        'qwen_description', '').lower():
        return 'Documentary'
    if 'Aerial Shot' in tags or 'Drone Footage' in tags:
        return 'Cinematic'
    return 'Unclassified'

```

# File: workflow.py

## Summary

This Python module provides a set of utility functions for handling video URLs, extracting video links from HTML content, downloading videos, and processing videos through a defined pipeline. It is designed to facilitate video processing tasks within a larger application, potentially serving as a backend service for video management.

## Changes Made

- Added function `is_video_url` to check if a URL is a direct video link.
- Added function `extract_video_url_from_html` to parse HTML content and extract video URLs.
- Added function `download_video` to download videos from URLs and save them to files.
- Added function `process_video_pipeline` to execute a video processing pipeline with customizable parameters.

```python
import logging
from core import setup_logging, load_config, ensure_directory, query_qwen_vl_chat, QwenVLModel
from analysis import extract_metadata, detect_scenes
from tagging import apply_tags, classify_video
from export import export_to_json
import os
import requests
from tqdm import tqdm
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import re


def is_video_url(url):
    """Checks if a given URL directly links to a video file.

Args:
    url (Any): The URL to check. This can be a string or any type that can be converted to a string.

Returns:
    Any: True if the URL is a direct video link, False otherwise. The return type can vary based on the implementation, but typically it will be a boolean value."""
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv',
        '.webm', '.ogg']
    if any(url.endswith(ext) for ext in video_extensions):
        return True
    try:
        response = requests.head(url, allow_redirects=True)
        content_type = response.headers.get('Content-Type', '')
        if 'video' in content_type:
            return True
    except requests.RequestException as e:
        logging.error(f'Failed to check URL content type: {e}')
    return False


def extract_video_url_from_html(html_content):
    """Extracts a video URL from HTML content.

Args:
    html_content (Any): The HTML content to parse. This can be a string or any type that can be processed as HTML.

Returns:
    Any: The extracted video URL as a string, or None if no URL is found. The return type may vary based on the implementation, but typically it will be a string or None."""
    soup = BeautifulSoup(html_content, 'html.parser')
    video_tag = soup.find('video')
    if video_tag:
        video_url = video_tag.get('src')
        if video_url:
            return video_url
    source_tag = soup.find('source')
    if source_tag:
        video_url = source_tag.get('src')
        if video_url:
            return video_url
    patterns = [
        '(?:"|\\\')((http[s]?:\\/\\/.*?\\.mp4|\\.mkv|\\.webm|\\.ogg))(?:\\"|\\\')'
        , '(?:"|\\\')((http[s]?:\\/\\/.*?video.*?))(?:\\"|\\\')']
    for pattern in patterns:
        match = re.search(pattern, html_content)
        if match:
            return match.group(1)
    return None


def download_video(url, filename):
    """Downloads a video from a URL and saves it to a file.

Args:
    url (Any): The URL of the video to download. This should be a string representing a valid video URL.
    filename (Any): The name to save the video as. This should be a string representing the desired file name, including the file extension.

Returns:
    Any: The path to the downloaded video file as a string, or None if the download fails. The return type may vary based on the implementation, but typically it will be a string or None."""
    config = load_config()
    download_dir = config.get('Paths', 'DownloadDirectory', fallback=
        'downloaded_videos')
    ensure_directory(download_dir)
    file_path = os.path.join(download_dir, filename)
    logging.info(f'Checking if URL {url} is a direct video link')
    if is_video_url(url):
        logging.info(f'URL is a direct video link, downloading from {url}')
        direct_url = url
    else:
        logging.info(
            f'URL {url} is not a direct video link, attempting to scrape the page for videos.'
            )
        try:
            response = requests.get(url)
            response.raise_for_status()
            if 'html' in response.headers.get('Content-Type', ''):
                direct_url = extract_video_url_from_html(response.text)
                if not direct_url:
                    logging.error(
                        f'Could not find a video link in the provided URL: {url}'
                        )
                    return None
            else:
                logging.error(
                    f'Content from URL {url} does not appear to be HTML.')
                return None
        except requests.exceptions.RequestException as e:
            logging.error(f'Error during request: {e}')
            return None
    try:
        response = requests.get(direct_url, stream=True)
        response.raise_for_status()
        with open(file_path, 'wb') as f, tqdm(total=int(response.headers.
            get('content-length', 0)), unit='iB', unit_scale=True,
            unit_divisor=1024) as progress_bar:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                progress_bar.update(size)
        logging.info(f'Video downloaded successfully: {filename}')
        return file_path
    except Exception as e:
        logging.error(f'Failed to download video: {e}', exc_info=True)
        return None


def process_video_pipeline(video_url, scene_threshold=None, custom_rules=
    None, output_dir=None):
    """Executes the full processing pipeline for a single video.

Args:
    video_url (Any): The URL of the video to process. This can be a string or any type that can be converted to a string.
    scene_threshold (Any): A threshold value used for scene detection. The type can vary based on the implementation, typically a float or integer.
    custom_rules (Any): Custom processing rules that may be applied during the video processing. This could be a dictionary or any other structure depending on the rules defined.
    output_dir (Any): The directory where the processed video will be saved. This should be a string representing a valid directory path.

Returns:
    Any: The result of the processing pipeline, which could vary based on the implementation. Typically, it may return a status message, a path to the output file, or None if processing fails."""
    config = load_config()
    scene_threshold = (scene_threshold if scene_threshold is not None else
        config.getfloat('SceneDetection', 'DefaultThreshold', fallback=0.3))
    output_dir = output_dir if output_dir is not None else config.get('Paths',
        'ProcessedDirectory', fallback='processed_videos/')
    logging.info(f'Starting pipeline for video: {video_url}')
    ensure_directory(output_dir)
    filename = os.path.basename(urlparse(video_url).path)
    local_filepath = download_video(video_url, filename)
    if local_filepath:
        metadata = extract_metadata(local_filepath)
        if metadata:
            scenes = detect_scenes(local_filepath, threshold=scene_threshold)
            qwen_model = QwenVLModel()
            qwen_description = qwen_model.process_video(local_filepath,
                metadata)
            video_info = {**metadata, 'qwen_description': qwen_description,
                'scenes': scenes}
            tags = apply_tags(video_info, custom_rules=custom_rules)
            classification = classify_video(video_info)
            video_info.update({'tags': tags, 'classification': classification})
            logging.info(
                f'Video classified as {classification} with tags: {tags}')
            export_to_json(video_info, output_dir=output_dir)
        else:
            logging.error(f'Metadata extraction failed for {local_filepath}.')
    else:
        logging.error(f'Failed to download video from URL: {video_url}')

```

