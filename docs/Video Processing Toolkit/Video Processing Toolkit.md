### **`utils.py`**

```python
import os
import logging
import requests
import validators
import hashlib

def ensure_directory(directory):
    """Ensures that the specified directory exists. Creates it if it doesn't."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        logging.info(f"Created directory: {directory}")
    else:
        logging.debug(f"Directory exists: {directory}")

def is_valid_url(url):
    """Checks if a URL is valid using the `validators` library."""
    return validators.url(url)

def get_final_url(initial_url):
    """Follows redirects to get the final URL of a resource."""
    try:
        response = requests.head(initial_url, allow_redirects=True, timeout=10)
        response.raise_for_status()
        return response.url
    except requests.exceptions.RequestException as e:
        logging.error(f"Error getting final URL for {initial_url}: {e}")
        return None

def calculate_video_hash(filepath, algorithm='sha256'):
    """Calculates the hash of a video file using the specified algorithm."""
    hash_obj = hashlib.new(algorithm)
    try:
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_obj.update(chunk)
        return hash_obj.hexdigest()
    except FileNotFoundError:
        logging.error(f"File not found: {filepath}")
        return None
```

### **`core.py`**

```python
import requests
import configparser
import logging
import os
import json
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from logging.handlers import RotatingFileHandler
from pymongo.errors import ConnectionFailure
# Initial configurations and logging setup
config = configparser.ConfigParser()
config.read('config.ini')  # Load configuration from config.ini

def load_config(config_path='config.ini'):
    """
    Loads the configuration from the specified config file.

    Args:
        config_path (str, optional): Path to the configuration file. Defaults to 'config.ini'.

    Returns:
        configparser.ConfigParser: The loaded configuration object.
    """
    try:
        config.read(config_path)
        return config
    except configparser.Error as e:
        logging.error(f"Error reading {config_path}: {e}")
        raise  # Re-raise the exception to halt execution

def setup_logging(log_file='video_processing.log', log_level=logging.INFO):
    """Sets up the logging configuration."""
    log_handler = RotatingFileHandler(log_file, maxBytes=1024*1024, backupCount=5)
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[log_handler])

def connect_to_mongodb():
    """Establishes a connection to the MongoDB database."""
    try:
        db_uri = config.get("MongoDB", "URI")
        client = MongoClient(db_uri, server_api=ServerApi('1'))
        client.admin.command('ping')  # Test the connection
        logging.info("Connected to MongoDB successfully.")
        return client
    except ConnectionFailure as e:
        logging.error(f"MongoDB connection failed: {e}")
        return None

def get_mongodb_collection():
    """
    Retrieves the MongoDB collection specified in the config file.

    Returns:
        pymongo.collection.Collection: The MongoDB collection object if successful, 
                                        otherwise None.
    """
    try:
        client = connect_to_mongodb()
        if client:
            database_name = config.get("MongoDB", "DatabaseName", fallback="video_database")
            collection_name = config.get("MongoDB", "CollectionName", fallback="videos")
            db = client[database_name]
            collection = db[collection_name]
            return collection
        else:
            return None
    except Exception as e:
        logging.error(f"Could not retrieve MongoDB collection: {e}")
        return None

def load_priority_keywords(file_path='priority_keywords.json'):
    """
    Loads priority keywords from a JSON file.

    Args:
        file_path (str, optional): Path to the JSON file containing priority keywords. 
                                    Defaults to 'priority_keywords.json'.

    Returns:
        list: A list of priority keywords, or an empty list if the file is not found or 
              an error occurs.
    """
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as file:
                return json.load(file)
        except Exception as e:
            logging.error(f"Could not load priority keywords: {e}")
    return []

def query_qwen_vl_chat(video_url, instruction="Describe this video.", use_vpc=False):
    """
    Sends a request to the Qwen-VL-Chat model API over HTTPS to process a video.

    Args:
        video_url (str): The URL of the video file.
        instruction (str): The instruction or query to be passed to the AI model.
        use_vpc (bool): Whether to use the VPC endpoint instead of the public one.

    Returns:
        str: The model's response, typically a description or analysis of the video.
    """
    # Load from config with fallbacks
    public_endpoint = config.get("QwenVL", "PublicEndpoint", fallback="https://your-public-endpoint")
    vpc_endpoint = config.get("QwenVL", "VpcEndpoint", fallback="https://your-vpc-endpoint")
    access_key = config.get("QwenVL", "AccessKey", fallback="your-access-key")

    # Choose the endpoint depending on VPC flag
    endpoint = vpc_endpoint if use_vpc else public_endpoint

    # Add priority keywords to the instruction
    priority_keywords = load_priority_keywords()
    if priority_keywords:
        instruction += " Focus on these aspects: " + ", ".join(priority_keywords)

    # Prepare the request payload
    payload = {
        "video_url": video_url,
        "instruction": instruction,
    }

    # Set the Authorization header with the access key.
    headers = {
        "Authorization": f"Bearer {access_key}",
        "Content-Type": "application/json"
    }

    try:
        # Send a POST request to the model's API over HTTPS
        response = requests.post(endpoint, json=payload, headers=headers)

        # Handle the response and return the description from the AI model
        if response.status_code == 200:
            result = response.json()
            return result.get("description", "No description available.")
        else:
            logging.error(f"API Error {response.status_code}: {response.text}")
            return None

    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to connect to the API: {e}")
        return None
```

### **`main.py`**

```python
import os
import logging
import configparser
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from core import setup_logging, load_config, ensure_directory
from database import create_database
from qwen_model import QwenVLModel
from workflow import process_video_pipeline
from interface import parse_arguments, read_video_metadata_from_csv

# Load configuration
config = configparser.ConfigParser()
try:
    config.read('config.ini')
    DOWNLOAD_DIRECTORY = config.get('Paths', 'DownloadDirectory', fallback='downloaded_videos')
    DATABASE_FILE = config.get('Paths', 'DatabaseFile', fallback='video_index.db')
    DROPBOX_ACCESS_TOKEN = config.get('Dropbox', 'AccessToken', fallback=None)
    MAX_WORKERS = config.getint('Concurrency', 'MaxWorkers', fallback=5)
    QWEN_MODEL_DIR = config.get('Qwen', 'ModelDir', fallback='/mnt/model/Qwen2-VL-7B-Instruct')
    MONGODB_URI = config.get('MongoDB', 'URI')
except configparser.Error as e:
    print(f"Error reading config.ini: {e}")
    exit(1)

# Database lock for thread safety
db_lock = threading.Lock()

# Global Qwen2-VL model instance
qwen_model = None

def download_and_process_videos(urls, qwen_model_instance):
    """Downloads and processes videos using multithreading."""
    logging.info("Starting download and processing of videos.")
    results = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit tasks to the executor and store the URL with the future
        futures = {executor.submit(process_video_pipeline, url, qwen_model_instance): url for url in urls}

        # Process results as they become available
        for future in as_completed(futures):
            url = futures[future]  # Get the URL associated with the future
            try:
                success = future.result()
                results.append((future.result(), success, url))  # Store result, success flag, and URL
                if success:
                    logging.info(f"{url} processing completed successfully.")
                else:
                    logging.warning(f"{url} processing encountered an error.")
            except Exception as exc:
                logging.error(f"{url} generated an exception: {exc}", exc_info=True)
                results.append((None, False, url))  # Store None, failure flag, and URL for errors

    return results

def main():
    """Main function to handle video downloads and uploads."""
    logging.debug("Starting main function.")
    if not os.path.exists(DOWNLOAD_DIRECTORY):
        os.makedirs(DOWNLOAD_DIRECTORY)
        logging.debug(f"Created download directory: {DOWNLOAD_DIRECTORY}")

    create_database()

    # Load Qwen2-VL model
    qwen_model = QwenVLModel(model_dir=QWEN_MODEL_DIR)

    # Parse command-line arguments
    args = parse_arguments()
    setup_logging(log_file='video_processing.log', log_level=getattr(logging, args.log_level))

    # Get video URLs from user input or a CSV file
    if args.csv:
        logging.info(f"Reading videos from CSV file: {args.csv}")
        videos = read_video_metadata_from_csv(args.csv)
        urls = [video['public_url'] for video in videos]
    else:
        urls = args.urls

    if urls:
        download_and_process_videos(urls, qwen_model)
    else:
        logging.info("No URLs entered. Exiting.")

if __name__ == "__main__":
    main()
```

### **`video_processing.py`**

```python
import os
import logging
from urllib.parse import urlparse
from tqdm import tqdm
import requests
from bs4 import BeautifulSoup
import re
from gridfs import GridFS
from core import connect_to_mongodb, config, ensure_directory

def is_video_url(url):
    """Checks if a URL directly links to a video file."""
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.ogg']
    if any(url.endswith(ext) for ext in video_extensions):
        return True

    try:
        response = requests.head(url, allow_redirects=True)
        content_type = response.headers.get('Content-Type', '')
        if content_type.startswith('video/'):
            return True
    except requests.RequestException as e:
        logging.error(f"Failed to check URL content type: {e}")

    return False

def extract_video_url_from_html(html_content):
    """
    Extracts a video URL from HTML content, including URLs embedded in JavaScript.

    Args:
        html_content (str): The HTML content to parse.

    Returns:
        str or None: The extracted video URL, or None if no URL is found.
    """
    from bs4 import BeautifulSoup
    import re

    soup = BeautifulSoup(html_content, 'html.parser')

    # 1. Check standard <video> and <source> tags
    video_tag = soup.find('video')
    if video_tag:
        video_url = video_tag.get('src')
        if video_url:
            return video_url

    source_tags = soup.find_all('source')
    for source_tag in source_tags:
        video_url = source_tag.get('src')
        if video_url:
            return video_url

    # 2. Search for URLs in <script> tags (JavaScript)
    script_tags = soup.find_all('script')
    for script_tag in script_tags:
        script_content = script_tag.string
        if script_content:
            # Look for common patterns for video URLs in JavaScript
            video_urls = extract_video_urls_from_javascript(script_content)
            if video_urls:
                return video_urls[0]  # Return the first found URL

    # 3. Search for URLs in other elements (e.g., data attributes)
    # You can add more custom logic here based on the specific websites you're scraping

    return None

def extract_video_urls_from_javascript(script_content):
    """
    Extracts video URLs from JavaScript code using regular expressions.

    Args:
        script_content (str): The JavaScript code to search.

    Returns:
        list: A list of extracted video URLs.
    """
    video_urls = []
    # Define a list of regular expressions to match different video URL patterns
    patterns = [
        r'(https?:\/\/.*?\.(mp4|webm|ogg|mov))',  # Match common video file extensions
        r'(https?:\/\/.*?\/videos\/.*?)',  # Match URLs containing "/videos/"
        r'(https?:\/\/.*?player\.vimeo\.com\/video\/.*?)',  # Match Vimeo URLs
        r'(https?:\/\/(?:www\.)?youtube\.com\/watch\?v=.*?)',  # Match standard YouTube video URLs
        r'(https?:\/\/youtu\.be\/.*?)',  # Match shortened YouTube URLs
        r'(https?:\/\/(?:www\.)?youtube\.com\/embed\/.*?)',  # Match embedded YouTube player URLs
        r'(https?:\/\/(?:www\.)?dailymotion\.com\/video\/.*?)',  # Match Dailymotion video URLs
        r'(https?:\/\/player\.vimeo\.com\/video\/.*?)',  # Match embedded Vimeo player URLs
        r'(https?:\/\/(?:www\.)?vimeo\.com\/.*?)',  # Match standard Vimeo video URLs
        r'(https?:\/\/(?:www\.)?twitch\.tv\/videos\/.*?)',  # Match Twitch video URLs
        r'(https?:\/\/(?:www\.)?facebook\.com\/.*?\/videos\/.*?)',  # Match Facebook video URLs
        r'(https?:\/\/.*?\.(mp4|webm|ogg|mov|avi|flv|wmv))'  # Match common video file extensions
        # Add more patterns as needed for other video hosting services
    ]

    for pattern in patterns:
        matches = re.findall(pattern, script_content)
        video_urls.extend(matches)

    return video_urls

def download_with_retry(url, max_retries=3, backoff_factor=0.3):
    """
    Downloads content from a URL with a retry mechanism.

    Args:
        url (str): The URL to download from.
        max_retries (int, optional): The maximum number of retries. Defaults to 3.
        backoff_factor (float, optional): The factor to increase the delay between retries. Defaults to 0.3.

    Returns:
        requests.Response: The response object if the download is successful.

    Raises:
        requests.exceptions.RequestException: If the download fails after all retries.
    """
    for attempt in range(max_retries):
        try:
            response = requests.get(url, stream=True, timeout=10)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                raise
            logging.warning(f"Retrying download for {url} (attempt {attempt + 1}): {e}")
            time.sleep(backoff_factor * (2 ** attempt))

def download_video(url, filename, download_dir=None, store_in_mongo=False):
    """
    Downloads a video from a URL and optionally stores it in MongoDB using GridFS.

    Args:
        url (str): The URL of the video to download.
        filename (str): The name to save the video as.
        download_dir (str, optional): The directory to save the video in. Defaults to the configured download directory.
        store_in_mongo (bool, optional): Whether to store the video in MongoDB using GridFS. Defaults to False.

    Returns:
        str or None: The path to the downloaded video file, or None if the download fails.
    """
    download_dir = download_dir or config.get('Paths', 'DownloadDirectory', fallback='downloaded_videos')
    ensure_directory(download_dir)
    file_path = os.path.join(download_dir, filename)

    logging.info(f"Checking if URL {url} is a direct video link")

    if is_video_url(url):
        logging.info(f"URL is a direct video link, downloading from {url}")
        direct_url = url
    else:
        logging.info(f"URL {url} is not a direct video link, attempting to scrape the page for videos.")
        try:
            response = requests.get(url)
            response.raise_for_status()
            if 'html' in response.headers.get('Content-Type', ''):
                direct_url = extract_video_url_from_html(response.text)
                if not direct_url:
                    logging.error(f"Could not find a video link in the provided URL: {url}")
                    return None
            else:
                logging.error(f"Content from URL {url} does not appear to be HTML.")
                return None
        except requests.exceptions.RequestException as e:
            logging.error(f"Error during request: {e}")
            return None

    try:
        response = download_with_retry(direct_url)

        with open(file_path, 'wb') as f, tqdm(
            total=int(response.headers.get('content-length', 0)),
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as progress_bar:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                progress_bar.update(size)

        logging.info(f"Video downloaded successfully: {filename}")

        if store_in_mongo:
            db_client = connect_to_mongodb()
            fs = GridFS(db_client[config.get('MongoDB', 'DatabaseName', fallback='video_database')])
            with open(file_path, 'rb') as f:
                file_id = fs.put(f, filename=filename)
            logging.info(f"Video stored in MongoDB: {filename}, GridFS ID: {file_id}")
            return file_id, file_path

        return file_path
    except Exception as e:
        logging.error(f"Failed to download video: {e}", exc_info=True)
        return None

def convert_video_format(video_path, output_format='mp4'):
    """
    Converts a video to a specified format using FFmpeg.

    Args:
        video_path (str): The path to the input video file.
        output_format (str, optional): The desired output format (e.g., 'mp4', 'avi'). Defaults to 'mp4'.

    Returns:
        str or None: The path to the converted video file, or None if conversion fails.
    """
    try:
        output_path = os.path.splitext(video_path)[0] + f'.{output_format}'
        (
            ffmpeg
            .input(video_path)
            .output(output_path)
            .run(overwrite_output=True)
        )
        logging.info(f"Video converted to {output_format}: {output_path}")
        return output_path
    except ffmpeg.Error as e:
        logging.error(f"Error converting video: {e}")
        return None

def validate_video(video_path):
    """
    Validates a video file using FFmpeg to check for corruption.

    Args:
        video_path (str): The path to the video file.

    Returns:
        bool: True if the video is valid, False otherwise.
    """
    try:
        probe = ffmpeg.probe(video_path)
        # Check for basic video stream information
        if 'streams' in probe and any(stream['codec_type'] == 'video' for stream in probe['streams']):
            logging.info(f"Video validated: {video_path}")
            return True
        else:
            logging.warning(f"Invalid video: No video stream found in {video_path}")
            return False
    except ffmpeg.Error as e:
        logging.error(f"Error validating video: {e}")
        return False
```

### **`database.py`**

```python
import logging
import os
from datetime import datetime
from bson.objectid import ObjectId
from gridfs import GridFS
from pymongo import MongoClient, errors
from pymongo.server_api import ServerApi
from core import config, calculate_video_hash

def connect_to_mongodb():
    """Establishes a connection to the MongoDB database."""
    try:
        client = MongoClient(config.get('MongoDB', 'URI'), server_api=ServerApi('1'))
        client.admin.command('ping')
        logging.info("Connected to MongoDB successfully.")
        return client
    except errors.PyMongoError as e:
        logging.error(f"MongoDB connection failed: {e}")
        raise  # Raise the exception to halt execution if the database is critical

def get_video_collection():
    """Retrieves the MongoDB collection for videos."""
    try:
        client = connect_to_mongodb()
        db_name = config.get('MongoDB', 'DatabaseName', fallback='video_database')
        collection_name = config.get('MongoDB', 'CollectionName', fallback='videos')
        return client[db_name][collection_name]
    except Exception as e:
        logging.error(f"Could not retrieve MongoDB collection: {e}")
        return None

def create_database():
    """
    Connects to MongoDB and ensures a database and collection exist.

    Returns:
        pymongo.collection.Collection: The MongoDB collection object.
    """
    config = configparser.ConfigParser()
    config.read('config.ini')

    try:
        client = MongoClient(config.get('MongoDB', 'URI'), server_api=ServerApi('1'))
        db_name = config.get('MongoDB', 'DatabaseName', fallback='video_database')
        collection_name = config.get('MongoDB', 'CollectionName', fallback='videos')
        db = client[db_name]

        # Create collection if it doesn't exist
        collection = db[collection_name]

        logging.info(f"Connected to MongoDB and retrieved collection '{collection_name}'.")
        return collection
    except errors.PyMongoError as e:
        logging.error(f"Error connecting to MongoDB or accessing collection: {e}")
        raise  # Critical error, should stop execution if the database is critical

def insert_video(original_url, file_path, filename):
    """Inserts video information into the MongoDB database."""
    collection = get_video_collection()
    file_hash = calculate_video_hash(file_path)

    # Check for duplicates
    if collection.find_one({'file_hash': file_hash}):
        logging.warning(f"Video with hash {file_hash} already exists in the database.")
        return None

    # Store video in GridFS
    db_client = connect_to_mongodb()
    fs = GridFS(db_client[config.get('MongoDB', 'DatabaseName', fallback='video_database')])
    with open(file_path, 'rb') as f:
        file_id = fs.put(f, filename=filename)

    video_data = {
        'filename': filename,
        'original_url': original_url,
        'download_date': datetime.now(),
        'file_path': file_path,
        'processed': False,
        'file_hash': file_hash,
        'metadata': None,
        'scenes': [],
        'qwen_description': None,
        'tags': [],
        'classification': None,
        'video_file': file_id  # Store GridFS file ID
    }

    try:
        result = collection.insert_one(video_data)
        logging.info(f"Video indexed: {filename}, MongoDB ID: {result.inserted_id}")
        return result.inserted_id
    except Exception as e:
        logging.error(f"Error indexing video: {e}")
        return None

def update_video(video_id, update_data):
    """Updates a video document in the MongoDB database."""
    collection = get_video_collection()
    try:
        result = collection.update_one({'_id': video_id}, {'$set': update_data})
        if result.modified_count > 0:
            logging.info(f"Video {video_id} updated successfully.")
            return True
        else:
            logging.warning(f"No changes made to video {video_id}.")
            return False
    except Exception as e:
        logging.error(f"Error updating video {video_id}: {e}")
        return False

def find_video_by_url(url):
    """Finds a video in the database by its original URL."""
    collection = get_video_collection()
    try:
        return collection.find_one({'original_url': url})
    except Exception as e:
        logging.error(f"Error finding video by URL: {e}")
        return None

def update_database_with_qwen_description(file_id, description):
    """
    Updates the database with the Qwen2-VL description.

    Args:
        file_id (ObjectId): The MongoDB ObjectId of the video document to update.
        description (str): The description generated by the Qwen2-VL model.

    Returns:
        bool: True if the update was successful, False otherwise.
    """
    collection = get_video_collection()
    if collection is None:
        return False

    try:
        result = collection.update_one(
            {'_id': file_id},
            {'$set': {'processed': True, 'qwen_description': description}}
        )
        if result.modified_count > 0:
            logging.info(f"Qwen2-VL description updated for video ID: {file_id}")
            return True
        else:
            logging.warning(f"No changes made to video {file_id}.")
            return False
    except Exception as e:
        logging.error(f"Error updating Qwen2-VL description: {e}")
        return False
```

### **`qwen_model.py`**

```python
import os
import logging
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from modelscope import snapshot_download
from qwen_vl_utils import process_vision_info
import torch
from gridfs import GridFS
from database import update_video
from core import connect_to_mongodb, config
from PIL import Image
import io

class QwenVLModel:
    def __init__(self, model_dir="/path/to/your/qwen2-vl-model"):
        """Initializes the Qwen2-VL model and processor."""
        try:
            self.model_path = snapshot_download("qwen/Qwen2-VL-7B-Instruct", cache_dir=model_dir)
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_path, torch_dtype="auto", device_map="auto"
            )
            self.model.eval()  # Set the model to evaluation mode
            self.processor = AutoProcessor.from_pretrained(self.model_path)
            logging.info("Qwen2-VL model loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading Qwen2-VL model: {e}")
            raise  # Re-raise the exception to stop execution

    def process_video(self, video_id, instruction="Describe this video."):
        """Processes a video using the Qwen2-VL model."""
        db_client = connect_to_mongodb()
        fs = GridFS(db_client[config.get('MongoDB', 'DatabaseName', fallback='video_database')])

        # Retrieve video data from GridFS
        grid_out = fs.get(video_id)
        video_data = io.BytesIO(grid_out.read())

        # Extract frames from the video data
        video_frames = self._extract_video_frames(video_data)

        if video_frames is None:
            logging.error(f"Failed to extract frames for video ID: {video_id}")
            return None

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video_frames, "fps": 1},  # Assuming fps is 1
                    {"type": "text", "text": instruction},
                ],
            }
        ]

        try:
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda" if torch.cuda.is_available() else "cpu")

            generated_ids = self.model.generate(**inputs, max_new_tokens=128)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

            # Update MongoDB with Qwen2-VL description
            update_video(video_id, {'qwen_description': output_text[0]})

            return output_text[0]
        except Exception as e:
            logging.error(f"Error processing video with Qwen2-VL: {e}", exc_info=True)
            return None

    def _extract_video_frames(self, video_data, fps=1, max_frames=16):
        """Extracts frames from video data using MoviePy."""
        try:
            video = VideoFileClip(video_data)
            frames = []
            for i, frame in enumerate(video.iter_frames(fps=fps)):
                if max_frames and i >= max_frames:
                    break
                frames.append(Image.fromarray(frame.astype('uint8'), 'RGB'))
            video.close()
            return frames
        except Exception as e:
            logging.error(f"Error extracting video frames: {e}")
            return None

```

### **`workflow.py`**

```python
import logging
import os
from core import load_config, ensure_directory, calculate_video_hash
from analysis import extract_metadata, detect_scenes
from tagging import apply_tags, classify_video
from export import export_to_json
from database import find_video_by_url, update_video, insert_video
from qwen_model import QwenVLModel
from video_processing import download_video

def process_video_pipeline(video_url, scene_threshold=None, custom_rules=None, output_dir=None):
    """
    Processes a single video through the entire pipeline: download, metadata extraction,
    scene detection, Qwen2-VL processing, tagging, classification, and export.

    Args:
        video_url (str): The URL of the video to process.
        scene_threshold (float, optional): Threshold for scene detection. Defaults to None.
        custom_rules (str, optional): Path to a JSON file with custom tagging rules. Defaults to None.
        output_dir (str, optional): Directory to save the processed data. Defaults to None.
    """
    config = load_config()

    scene_threshold = scene_threshold if scene_threshold is not None else config.getfloat('SceneDetection', 'DefaultThreshold', fallback=0.3)
    output_dir = output_dir if output_dir is not None else config.get('Paths', 'ProcessedDirectory', fallback='processed_videos/')

    logging.info(f"Starting pipeline for video: {video_url}")
    ensure_directory(output_dir)

    # Create a filename based on the URL
    filename = os.path.basename(urlparse(video_url).path)

    # Check for duplicates
    existing_video = find_video_by_url(video_url)
    if existing_video:
        logging.warning(f"Video already exists in the database: {video_url}")
        return

    # Download the video and store in MongoDB
    file_id, local_filepath = download_video(video_url, filename, store_in_mongo=True)

    if file_id and local_filepath:
        # Extract metadata and update MongoDB
        metadata = extract_metadata(local_filepath, file_id)

        # Detect scenes and update MongoDB
        scenes = detect_scenes(local_filepath, file_id, threshold=scene_threshold)

        # Load the Qwen2-VL model
        qwen_model = QwenVLModel(model_dir=config.get('Qwen', 'ModelDir', fallback='/mnt/model/Qwen2-VL-7B-Instruct'))

        # Process video using Qwen2-VL and update MongoDB
        qwen_description = qwen_model.process_video(file_id)

        # Aggregate data for tagging and classification
        video_info = {'qwen_description': qwen_description, 'scenes': scenes, 'filename': filename, **metadata}

        # Apply tags and classification
        tags = apply_tags(video_info, custom_rules=custom_rules)
        classification = classify_video(video_info)

        # Update MongoDB with tags and classification
        update_data = {'tags': tags, 'classification': classification}
        update_video(file_id, update_data)

        logging.info(f"Video classified as {classification} with tags: {tags}")

        # Optional: Export data to JSON
        export_to_json(video_info, output_dir=output_dir)
    else:
        logging.error(f"Failed to download or process video from URL: {video_url}")

```

### **`analysis.py`**

```python
import subprocess
import logging
import csv
import json
import os
from core import query_qwen_vl_chat, ensure_directory
from moviepy.editor import VideoFileClip
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector  # For content-aware scene detection

def segment_video(input_video, output_dir, segment_length=15, max_resolution="720p", max_fps=30):
    """
    Segments a video into smaller clips with specified resolution and FPS limits.

    Args:
        input_video (str): Path to the input video file.
        output_dir (str): Directory to save the segmented clips.
        segment_length (int, optional): Length of each segment in seconds. Defaults to 15.
        max_resolution (str, optional): Maximum resolution for the output clips (e.g., "720p", "1080p"). Defaults to "720p".
        max_fps (int, optional): Maximum frame rate for the output clips. Defaults to 30.
    """
    try:
        ensure_directory(output_dir)

        # Determine scale filter based on desired resolution
        if max_resolution == "720p":
            scale_filter = "scale=trunc(min(iw\,1280)/2)*2:trunc(min(ih\,720)/2)*2"
        elif max_resolution == "1080p":
            scale_filter = "scale=trunc(min(iw\,1920)/2)*2:trunc(min(ih\,1080)/2)*2"
        else:
            scale_filter = "scale=trunc(min(iw\,1280)/2)*2:trunc(min(ih\,720)/2)*2"  # Default to 720p

        command = [
            'ffmpeg',
            '-i', input_video,
            '-vf', f'{scale_filter},fps={max_fps}',
            '-c:v', 'libx264',
            '-c:a', 'copy',
            '-f', 'segment',
            '-segment_time', str(segment_length),
            f'{output_dir}/clip_%03d.mp4'
        ]
        subprocess.run(command, check=True)
        logging.info(f"Video '{input_video}' segmented into {segment_length}-second clips.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error segmenting video: {e}")

def extract_metadata(filepath, video_id):  # Add video_id parameter
    """
    Extracts video metadata using FFmpeg and updates the MongoDB document.

    Args:
        filepath (str): The path to the video file.
        video_id (ObjectId): The MongoDB ID of the video document.

    Returns:
        dict: A dictionary containing the extracted metadata, or None if extraction fails.
    """
    metadata = {}
    try:
        probe = ffmpeg.probe(filepath)
        video_stream = next((s for s in probe['streams'] if s['codec_type'] == 'video'), None)
        if video_stream:
            duration = float(probe['format']['duration']) if probe['format'].get('duration') else None
            metadata = {
                'duration': duration,
                'bit_rate': int(probe['format']['bit_rate']),
                'resolution': f"{video_stream['width']}x{video_stream['height']}",
                'frame_rate': video_stream['r_frame_rate'],
                'codec': video_stream['codec_name'],
                'size': os.path.getsize(filepath)
            }
        else:
            logging.warning(f"No video stream found in {filepath}")

    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to extract metadata: {e}")

    # Update MongoDB document with metadata
    update_video(video_id, {'metadata': metadata})
    return metadata

def detect_scenes(filepath, video_id, threshold=30.0):  # Add video_id parameter
    """
    Detects scene changes in a video using the `scenedetect` library and updates the MongoDB document.

    Args:
        filepath (str): The path to the video file.
        video_id (ObjectId): The MongoDB ID of the video document.
        threshold (float, optional): Sensitivity threshold for scene detection. Defaults to 30.0.

    Returns:
        list: A list of dictionaries, each containing the start and end time of a scene.
    """
    scene_list = []
    try:
        video_manager = VideoManager([filepath])
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=threshold))

        video_manager.set_downscale_factor()
        video_manager.start()
        scene_manager.detect_scenes(frame_source=video_manager)

        scenes = scene_manager.get_scene_list()
        scene_list = [
            {'start_time': scene[0].get_seconds(), 'end_time': scene[1].get_seconds()}
            for scene in scenes
        ]
        logging.info(f"Detected {len(scene_list)} scenes in {filepath}")

    except Exception as e:
        logging.error(f"Scene detection failed: {e}")

    # Update MongoDB document with scenes
    update_video(video_id, {'scenes': scene_list})
    return scene_list

def extract_scene_segment(video_path, start_time, end_time, output_dir="scene_segments"):
    """Extracts a scene segment from a video using MoviePy."""
    ensure_directory(output_dir)
    filename = os.path.basename(video_path).split('.')[0]  # Get filename without extension
    segment_path = os.path.join(output_dir, f"{filename}_{start_time:.2f}-{end_time:.2f}.mp4")
    try:
        video = VideoFileClip(video_path)
        segment = video.subclip(start_time, end_time)
        segment.write_videofile(segment_path, codec='libx264', audio_codec='aac')
        video.close()
        logging.info(f"Extracted scene segment: {segment_path}")
        return segment_path
    except Exception as e:
        logging.error(f"Error extracting scene segment: {e}")
        return None
```

### **`tagging.py`**

```python
import logging
import json
import os

def load_custom_tags(custom_rules_path):
    """Loads custom tagging rules from a JSON file."""
    if not custom_rules_path or not os.path.isfile(custom_rules_path):
        logging.warning(f"Custom tag rules file {custom_rules_path} not found.")
        return {}
    with open(custom_rules_path, 'r') as file:
        return json.load(file)

def apply_tags(video_info, custom_rules=None):
    """
    Applies tags to a video based on its metadata, AI-generated description, and custom rules.

    Args:
        video_info (dict): A dictionary containing video metadata and AI-generated description.
        custom_rules (str or None): Path to the custom rules JSON file.

    Returns:
        list: List of applied tags.
    """
    tags = []

    # Apply default tagging rules based on metadata
    if 'resolution' in video_info and '4K' in video_info['resolution']:
        tags.append("High Resolution")

    if 'duration' in video_info and video_info['duration'] > 600:
        tags.append("Extended Play")

    if video_info.get('codec') == "h264":
        tags.append("H.264 Codec")

    # Analyze AI-generated description for additional tags
    description = video_info.get('qwen_description', "").lower()

    if "action" in description:
        tags.append("Action")

    if "night shot" in description:
        tags.append("Night-time Filming")

    # (Add more rules/keywords as necessary)

    # Apply custom rules if available
    if custom_rules:
        custom_tags_rules = load_custom_tags(custom_rules)
        tags = apply_custom_rules(video_info, tags, custom_tags_rules)

    return list(set(tags))  # Remove duplicates

def apply_custom_rules(video_info, tags, custom_tags_rules):
    """
    Applies custom rules provided in a JSON file to tag the video.

    Args:
        video_info (dict): A dictionary containing video metadata and AI-generated descriptions.
        tags (list): List of already applied tags during the default rules processing.
        custom_tags_rules (dict): Dictionary of custom rules.

    Returns:
        list: Updated list of tags after applying custom rules.
    """
    rules = custom_tags_rules.get("rules", [])
    for rule in rules:
        if 'description_keywords' in rule:
            keywords = set(rule.get('description_keywords', []))
            if keywords.intersection(set(video_info.get('qwen_description', "").split())):
                tags.extend(rule.get('tags', []))

        if 'metadata_conditions' in rule:
            conditions = rule.get('metadata_conditions', {})
            resolution = video_info.get('resolution', "")
            duration = video_info.get('duration', 0)
            if (conditions.get('resolution') == resolution and
                duration > conditions.get('duration_gt', 0)):
                tags.extend(rule.get('tags', []))

    return list(set(tags))  # Remove duplicates

def classify_video(video_info):
    """
    Classifies a video into a high-level category based on its tags and metadata.

    Args:
        video_info (dict): A dictionary containing tagged video metadata.

    Returns:
        str: Classification of the video (e.g., "Action", "Documentary").
    """
    tags = video_info.get('tags', [])

    if "Action" in tags or "High Intensity" in tags:
        return "Action"

    if "Documentary Footage" in tags or "Narrative" in video_info.get('qwen_description', "").lower():
        return "Documentary"

    if "Aerial Shot" in tags or "Drone Footage" in tags:
        return "Cinematic"

    # (Add more conditional classification logic as needed)

    return "Unclassified"  # Default classification if no rules apply

```

### **`export.py`**

```python
import json
import logging
import os
from core import ensure_directory

def export_to_json(data, output_dir):
    """
    Export processed video data to a JSON file.

    Args:
        data (dict): Processed video data including metadata, tags, and classification.
        output_dir (str): Directory where the JSON file should be exported.
    """
    ensure_directory(output_dir)  # Ensure the output directory exists

    # Create the output file path using the video's filename
    filename = os.path.join(output_dir, f"{data['filename']}.json")

    try:
        # Write the data to a JSON file with pretty-print formatting
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        logging.info(f"Data exported to {filename}")
    except Exception as e:
        logging.error(f"Failed to export data to {filename}: {e}")
```

### **`interface.py`**

```python
import argparse
import logging
from core import load_config, ensure_directory, setup_logging, query_qwen_vl_chat
from analysis import segment_video  # Import the updated function
from workflow import process_video_pipeline
import json
import csv
import os

def csv_to_json(csv_file, output_json_file, max_resolution="720p", max_fps=30):
    """Converts CSV to JSON with resolution and FPS limits."""
    videos = []
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            video_data = {
                "file_name": row['file_name'],
                "file_size": row['file_size'],
                "last_modified": row['last_modified'],
                "public_url": row['public_url'],
                "segments": [],
                "max_resolution": max_resolution,
                "max_fps": max_fps
            }
            videos.append(video_data)

    for video in videos:
        output_dir = os.path.join("segmented_videos", video['file_name'])
        segment_video(video['public_url'], output_dir, 15, video['max_resolution'], video['max_fps'])
        for i in range(1, 999):
            segment_url = f"{video['public_url'][:-4]}_clip_{i:03d}.mp4"
            if os.path.exists(os.path.join(output_dir, f"clip_{i:03d}.mp4")):
                video['segments'].append(segment_url)
            else:
                break

    with open(output_json_file, 'w') as json_file:
        json.dump(videos, json_file, indent=2)
        print(f"JSON output successfully written to {output_json_file}")

def run_inference_on_videos(videos):
    """Runs inference on video segments, handling resolution and FPS."""
    all_results = []
    for video in videos:
        file_name = video['file_name']
        segment_results = []
        for segment_url in video['segments']:
            description = query_qwen_vl_chat(
                segment_url,
                instruction="Describe this video segment.",
                max_resolution=video['max_resolution'],
                max_fps=video['max_fps']
            )
            segment_results.append({
                'segment_url': segment_url,
                'description': description
            })
        all_results.append({
            'file_name': file_name,
            'segments': segment_results
        })
    return all_results

def parse_arguments():
    """
    Parses command-line arguments provided by the user.

    Returns:
        Namespace: Parsed command-line arguments.
    """
    config = load_config()  # Load configuration from the config.ini file

    parser = argparse.ArgumentParser(description='Comprehensive Video Processing Toolkit')

    # Set default values from the config file
    default_output = config.get('Paths', 'ProcessedDirectory', fallback='processed_videos/')
    default_scene_threshold = config.getfloat('SceneDetection', 'DefaultThreshold', fallback=0.3)
    default_qwen_instruction = config.get('QwenVL', 'DefaultInstruction', fallback="Describe this video.")

    parser.add_argument('--urls', nargs='+', help='List of video URLs to process')
    parser.add_argument('--csv', type=str, help='Path to the CSV file containing video URLs and metadata')  # New option for CSV input
    parser.add_argument('--config', type=str, help='Path to the configuration file', default='config.ini')
    parser.add_argument('--output', type=str, help='Output directory', default=default_output)
    parser.add_argument('--log_level', type=str, help='Set the logging level',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], default='INFO')

    # Processing options
    parser.add_argument('--scene_threshold', type=float, help='Threshold for scene detection', default=default_scene_threshold)
    parser.add_argument('--qwen_instruction', type=str, help='Instruction passed to Qwen-VL', default=default_qwen_instruction)
    parser.add_argument('--use_vpc', action='store_true', help='Use the VPC endpoint for Qwen-VL API calls')

    args = parser.parse_args()

    # Ensure they provided either URLs or a CSV, but not both
    if not args.urls and not args.csv:
        parser.error("You must provide either --urls or --csv to indicate which videos to process.")
    return args

def read_video_metadata_from_csv(csv_file):
    """
    Reads video metadata from a CSV file created with rclone.

    Args:
        csv_file (str): Path to the CSV file containing video metadata.

    Returns:
        list: A list of dictionaries containing video metadata (e.g., filename, file_size, public_url).
    """
    videos = []

    # Attempt to read the CSV file
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)

        # Validate that the necessary columns exist in the CSV
        required_columns = ['file_name', 'file_size', 'last_modified', 'public_url']
        for col in required_columns:
            if col not in reader.fieldnames:
                logging.error(f"The CSV file must contain the following columns: {', '.join(required_columns)}")
                return []

        # Read each row and append the video metadata
        for row in reader:
            # Check that the public URL is present
            if row.get('public_url'):
                videos.append({
                    'file_name': row.get('file_name'),
                    'file_size': row.get('file_size'),
                    'last_modified': row.get('last_modified'),
                    'public_url': row.get('public_url')
                })

    return videos

def main():
    """Main function to handle video processing."""
    args = parse_arguments()
    setup_logging(log_file='video_processing.log')
    logging.getLogger().setLevel(args.log_level)
    config = load_config(args.config)
    ensure_directory(args.output)

    if args.csv:
        logging.info(f"Reading videos from CSV file: {args.csv}")
        # Get max resolution and FPS from arguments or config
        max_resolution = args.max_resolution or config.get('Processing', 'MaxResolution', fallback='720p')
        max_fps = args.max_fps or config.getint('Processing', 'MaxFPS', fallback=30)
        csv_to_json(args.csv, "videos_with_segments.json", max_resolution, max_fps)
        with open("videos_with_segments.json", 'r') as f:
            videos = json.load(f)
        inference_results = run_inference_on_videos(videos)
        # ... (Process inference_results as needed, e.g., save to file)
    else:
        # Use manual URLs provided via --urls option
        videos = [{'public_url': url} for url in args.urls]

        # Iterate over videos and process each one
        for video_info in videos:
            video_url = video_info.get('public_url')
            filename = video_info.get('file_name', None)  # Optional for logging

            logging.info(f"Processing video: {filename} from URL: {video_url}")

            # Run the full processing pipeline for the given video
            process_video_pipeline(
                video_url=video_url,
                scene_threshold=args.scene_threshold,
                custom_rules=None,  # Optionally load custom rules if needed
                output_dir=args.output
            )

if __name__ == '__main__':
    main()
```

### **`config.ini`**

```ini
[Paths]
# Directory where downloaded videos will be stored
DownloadDirectory = downloaded_videos

# Directory where processed videos and metadata will be exported
ProcessedDirectory = processed_videos

[MongoDB]
URI = mongodb+srv://hperkin4:LVJJSbh6gp9nw1wt@cluster.qamnd.mongodb.net/?retryWrites=true&w=majority&appName=Cluster
DatabaseName = video_database  # Replace with your actual database name
CollectionName = videos

[Concurrency]
# Maximum number of workers (threads/processes) to use for parallel processing
MaxWorkers = 5

[QwenVL]
# Public endpoint for accessing the Qwen2-VL-Chat service
PublicEndpoint = https://quickstart-endpoint.region.aliyuncs.com/api/v1/inference

# VPC endpoint for accessing the Qwen2-VL-Chat service (use for deployments within the same VPC)
VpcEndpoint = https://quickstart-vpc-endpoint.region.aliyuncs.com/api/v1/inference

# Access key required for API authentication
AccessKey = Your_Access_Key

# Default instruction/query passed to the Qwen-VL API
DefaultInstruction = Describe this video.

[SceneDetection]
# Default threshold value for detecting scene changes in videos
DefaultThreshold = 0.3

[Rules]
# Default path for loading custom tagging rules (optional)
CustomTagRules = custom_tags.json

[Logging]
# Log file location
LogFile = video_processing.log
```

### **`requirements.txt`**

```ini
beautifulsoup4==4.11.1
blinker==1.4
certifi==2024.8.30
charset-normalizer==2.1.1
click==8.1.7
colorama==0.4.4
command-not-found==0.3
configparser==5.2.0
cryptography==3.4.8
dbus-python==1.2.18
decorator==4.4.2
distro==1.7.0
distro-info==1.1+ubuntu0.2
ffmpeg-python==0.2.0
filelock==3.16.0
fsspec==2024.9.0
future==0.18.2
httplib2==0.20.2
huggingface-hub==0.24.6
idna==3.8
imageio==2.35.1
imageio-ffmpeg==0.5.1
importlib-metadata==4.6.4
jeepney==0.7.1
Jinja2==3.1.4
keyring==23.5.0
launchpadlib==1.10.16
lazr.restfulclient==0.14.4
lazr.uri==1.0.6
lxml==4.9.1
MarkupSafe==2.1.5
modelscope==1.18.0
mongoengine==0.26.0
more-itertools==8.10.0
moviepy==1.0.3
mpmath==1.3.0
netifaces==0.11.0
networkx==3.3
numpy==2.1.1
nvidia-cublas-cu12==12.1.3.1
nvidia-cuda-cupti-cu12==12.1.105
nvidia-cuda-nvrtc-cu12==12.1.105
nvidia-cuda-runtime-cu12==12.1.105
nvidia-cudnn-cu12==9.1.0.70
nvidia-cufft-cu12==11.0.2.54
nvidia-curand-cu12==10.3.2.106
nvidia-cusolver-cu12==11.4.5.107
nvidia-cusparse-cu12==12.1.0.106
nvidia-nccl-cu12==2.20.5
nvidia-nvjitlink-cu12==12.6.68
nvidia-nvtx-cu12==12.1.105
oauthlib==3.2.0
packaging==24.1
pillow==10.4.0
platformdirs==4.3.2
proglog==0.1.10
PyGObject==3.42.1
PyJWT==2.3.0
pymongo==4.2.0
pyparsing==2.4.7
python-apt==2.4.0+ubuntu4
PyYAML==6.0.2
regex==2024.9.11
requests==2.28.1
safetensors==0.4.5
scenedetect==0.6.4
SecretStorage==3.3.1
six==1.16.0
soupsieve==2.6
sympy==1.13.2
systemd-python==234
tokenizers==0.19.1
torch==2.4.1
tqdm==4.64.1
transformers==4.44.2
triton==3.0.0
typing_extensions==4.12.2
ubuntu-drivers-common==0.0.0
ubuntu-pro-client==8001
ufw==0.36.1
unattended-upgrades==0.1
urllib3==1.26.20
validators==0.34.0
wadllib==1.3.6
xkit==0.0.0
zipp==1.0.0
meson-python
ninja
meson

```

### **`README.md`**

## Comprehensive Video Processing Toolkit

## Overview

This toolkit automates downloading, analyzing, tagging, and classifying videos. It leverages the Qwen2-VL multimodal AI model for content understanding, along with traditional video processing techniques, to provide rich metadata and insights.

## Key Features

- **Video Download:** Downloads videos from various sources (direct links and HTML pages).
- **Metadata Extraction:** Extracts technical metadata using FFmpeg (duration, resolution, codecs, etc.).
- **Scene Detection:** Identifies scene changes for granular video analysis.
- **AI-Powered Analysis:**  Analyzes video content using the Qwen2-VL model, generating descriptions and insights.
- **Customizable Tagging:**  Applies tags based on metadata, Qwen2-VL output, and custom rules.
- **Video Classification:**  Classifies videos into categories (e.g., sports, music).
- **Data Export:** Exports processed data in JSON format.
- **MongoDB Integration (Optional):** Stores data in a MongoDB database.

## Prerequisites

- **Python 3.7 or higher:** [https://www.python.org/downloads/](https://www.python.org/downloads/)
- **FFmpeg:** Install FFmpeg on your system. Instructions vary by operating system.
- **MongoDB (Optional):**  If you want to use MongoDB for storage, install it and set up a database.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/henryperkins/video_processing_toolkit.git
   cd video_processing_toolkit
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the Qwen2-VL Model:**
   - Download the model files from Hugging Face: [https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct)
   - Extract the model files to the directory specified in your `config.ini` file (see below).

5. **Configure `config.ini`:**
   - Create a `config.ini` file in the project directory and customize the settings:

## Usage

1. **Run the script:**
   ```bash
   python interface.py --urls <video_url_1> <video_url_2> ... --output <output_directory>
   ```
   - Replace `<video_url_1>`, `<video_url_2>`, etc. with the URLs of the videos you want to process.
   - Replace `<output_directory>` with the directory where you want to save the processed JSON files.

2. **(Optional) Use a CSV file:**
   - Create a CSV file with columns `file_name`, `file_size`, `last_modified`, and `public_url` containing video information.
   - Run the script with the `--csv` option:
     ```bash
     python interface.py --csv <path_to_csv_file> --output <output_directory>
     ```

## Output

The toolkit generates a JSON file for each processed video in the specified output directory. The JSON file contains:

- Video metadata (duration, resolution, codec, etc.)
- Scene change timestamps
- Qwen2-VL generated description
- Applied tags
- Video classification

## Examples

- **Process videos from URLs:**
  ```bash
  python interface.py --urls https://www.example.com/video1.mp4 https://www.example.com/video2.mp4 --output processed_videos
  ```

- **Process videos from a CSV file:**
  ```bash
  python interface.py --csv video_list.csv --output processed_videos
  ```

## Contributing

Contributions are welcome! Please feel free to submit a pull request.

## License

This project is licensed under the MIT License.