> This document is the journal of a python script's progression from single file script to multi-file toolkit:

# Use Cases for the Script

**1. Building a Video Dataset for Action Recognition:**

* **Goal:** Create a labeled dataset of videos to train a machine learning model for recognizing specific actions or events.
* **How the Script Helps:**
	* **Automated Download:** Downloads videos from various sources based on provided URLs.
	* **Metadata Extraction:** Extracts relevant metadata (duration, resolution, etc.) for analysis and filtering.
	* **Tagging:** Applies tags based on filename, metadata, or potentially content analysis (using FFmpeg or Qwen2-VL). These tags can serve as initial labels for training.
	* **Qwen2-VL Description:** The model's description can provide additional insights into the video content, which can be used for labeling or refining tags.

**2. Content Analysis and Categorization:**

* **Goal:** Analyze a collection of videos to understand their content, identify trends, or categorize them based on specific criteria.
* **How the Script Helps:**
	* **Metadata and Tagging:** Provides structured information about each video, making it easier to search, filter, and analyze the dataset.
	* **Qwen2-VL Analysis:** The model's output can be used to automatically categorize videos based on their content, identify key themes, or extract relevant information.

**3. Building a Searchable Video Library:**

* **Goal:** Create a searchable database of videos with rich metadata and descriptions.
* **How the Script Helps:**
	* **Indexing:** Stores all video information in a structured database (MongoDB), making it easily searchable.
	* **Metadata and Tags:** Provides multiple ways to search and filter videos (by filename, duration, resolution, tags, etc.).
	* **Qwen2-VL Descriptions:** Allows users to search for videos based on their content or themes identified by the model.

---

# High-Level Overview of the Current Script

## **Current State of the Script:**

The script you've been developing is a comprehensive, modular video processing toolkit designed to serve three primary use cases:

1) **Building a Video Dataset for Action Recognition:**
   * **Goal**: Automate the process of downloading, tagging, and categorizing videos to create a labeled dataset for machine learning models focused on action recognition.

2) **Content Analysis and Categorization:**
   * **Goal**: Facilitate the analysis and categorization of video collections based on their content, through structured metadata and automatic tagging/classification.

3) **Building a Searchable Video Library:**
   * **Goal**: Allow users to search and filter through a vast collection of videos based on metadata, content, and AI-generated descriptions, creating a rich, searchable video library.

The script has transitioned from a simple monolithic script into a sophisticated multi-file toolkit, with each module focused on a specific responsibility, making the script scalable, maintainable, and easy to develop further.

## **Direction And Objectives:**
* **Modular Structure:** The script's structure allows components like downloading, tagging, analysis, and export to be isolated, developed independently, and extended over time.
* **AI Integration:** Through the integration of the Qwen2-VL-Chat model, the script is aimed at incorporating AI capabilities, particularly in content-based video descriptions for enhanced tagging and categorization.
* **Performance and Scalability:** The direction of the script emphasizes optimization, ease of configuration, and scalability, making it suitable for large datasets and potential future integrations (e.g., more advanced AI models or cloud-based storage).

---

# Detailed Breakdown of Key Functions/Modules

Let's break down each module and its key functions:

## 1. **`core.py`**
   * **Purpose:** Provides essential utility functions such as configuration loading, logging setup, directory management, and MongoDB connectivity.
   * **Key Functions:**
	 * **`load_config()`**: Loads and parses the `config.ini` file, providing access to configuration values across all modules.
	 * **`setup_logging()`**: Configures the logging mechanism, which is critical for monitoring the script's operations and debug any issues.
	 * **`ensure_directory()`**: Ensures that required directories exist; if not, they are created.
	 * **`connect_to_mongodb()`**: Establishes a connection to MongoDB (or Cosmos DB for MongoDB), validating the connection by sending a ping to the server.
	 * **`query_qwen_vl_chat()`**: Makes an HTTP request to the Qwen2-VL-Chat API, sending video URLs and instructions, and retrieving AI-generated descriptions for further processing.

## 2. **`workflow.py`**
   * **Purpose:** Orchestrates the overall video processing pipeline, integrating the functionality from other modules to download, analyze, tag, and export video data.
   * **Key Functions:**
	 * **`process_video_pipeline()`**: The centerpiece of the script, coordinating the workflow for each video, from downloading and analysis to exporting processed data.
	 * **`download_video()`**: Downloads a video from a provided URL and stores it locally, either for further processing or archival.
	 * **`detect_scenes()`**: Analyzes the video to detect and timestamp scene changes, aiding in detailed content analysis.
	 * **`extract_metadata()`**: Uses FFmpeg to extract video metadata like duration, resolution, codec, and audio channels, which is crucial for both indexing and analysis.
	 * **`work_with_qwenvl()`**: Enhances video metadata by adding AI-generated content descriptions, tags, and classifications.

## 3. **`analysis.py`**
   * **Purpose:** Handles video metadata extraction, scene detection, and Qwen2-VL processing for enhancing video metadata with AI-generated insights.
   * **Key Functions:**
	 * **`extract_metadata()`**: The function to extract and return detailed metadata from video files (like resolution, FPS, etc.), aiding thorough analysis.
	 * **`detect_scenes()`**: Leverages FFmpeg's scene detection to segment videos, storing important timestamps.
	 * **`process_with_qwenvl()`**: Submits videos for processing by the Qwen2-VL-Chat model and returns the AI-generated descriptions.

## 4. **`tagging.py`**
   * **Purpose:** Responsible for applying tags and classifications to videos based on extracted metadata, AI-generated descriptions, and predefined rule sets.
   * **Key Functions:**
	 * **`apply_tags()`**: Analyzes the metadata and Qwen-VL descriptions to assign relevant tags, allowing for easy categorization.
	 * **`classify_video()`**: Classifies videos into broad categories such as "Action," "Documentary," or "Sports" based on the content and tags.
	 * **`apply_custom_rules()`**: Allows for custom tagging rules, adding flexibility for users to define their specific use cases, supporting further extensions.

## 5. **`export.py`**
   * **Purpose:** Manages the final export of processed data, ensuring all video analysis, tagging, and classification results are systematically saved for later use.
   * **Key Functions:**
	 * **`export_to_json()`**: Saves the processed metadata (inclusive of tags, classification, and AI descriptions) in a structured JSON format, allowing easy access and further processing.

## 6. **`interface.py`**
   * **Purpose:** Provides a command-line interface (CLI) for interacting with the script, enabling users to specify processing parameters dynamically.
   * **Key Functions:**
	 * **`parse_arguments()`**: Handles CLI argument parsing, allowing users to customize output directories, instructions for the Qwen2-VL-Chat model, and other parameters.
	 * **`main()`**: Serves as the main entry point of the entire script, tying all modules together based on user inputs and running the video processing pipeline.

---

# Workflow Structure

Here's an outline-based workflow followed by a visual description:

1) **Initialization Phase:**
   * Load configuration settings from `config.ini`.
   * Set up logging to record actions and errors.

2) **Video Download and Metadata Extraction:**
   * The pipeline begins by downloading the video files from provided URLs (`workflow.py`).
   * Extract essential metadata from the downloaded video to prepare for analysis (`analysis.py`).

3) **Scene Detection and AI-Based Content Analysis:**
   * Detect scene changes in the video, creating segments for more granular analysis (e.g., action detection) (`analysis.py`).
   * Submit the video to the Qwen2-VL-Chat model for AI-based description and possibly action/event recognition (`workflow.py` using `core.py`).

4) **Tagging and Classification:**
   * Apply tags based on metadata, AI descriptions, and custom rules (`tagging.py`).
   * Automatically categorize videos into predefined classes such as "Action," "Sports," etc. (`tagging.py`).

5) **Exporting and Data Storage:**
   * Collate the metadata, tags, AI descriptions, and classification to form a complete structured dataset.
   * Export this dataset into a JSON file for easy indexing/searching in a video library or for training machine learning models (`export.py`).

---

# Visual Workflow Diagram

```html
[Initialization Phase]
    |
    |---> [core.py] --- Load Config & Logging
                         |
    +-----------------------------------------------
    |
    v
[Video Download & Metadata Extraction]              -----> [Process with Qwen2-VL]
    workflow.py
       |                                              |
       |--> [Download Video] --- [Extract Metadata]   |
          |                                            |
          |                                        Export JSON
          |
          v
        |
        v
[Scene Detection & AI-based Analysis]               ----- [Detect Scenes] - [Generate AI-based Descriptions]
    analysis.py [Analyze Video]
       |
       v
[Tagging & Classification] - [Apply Tags & Classify Videos]
    tagging.py
       |
       v
    |
    v
[Export Phase] ----> Export Results (JSON)
    export.py
```

The diagram outlines the flow from loading configuration to exporting processed video data, involving the key components from each module.

# Integrating the Script for the Three Primary Use Cases

The script is well-positioned to fulfill the three primary use cases:

1) **Building Video Datasets for Action Recognition:**
   * The script can automate bulk video downloads, tagging based on both metadata and AI descriptions, and export of labeled datasets, ready to be used in training ML models.

2) **Content Analysis and Categorization:**
   * By leveraging scene detection, metadata extraction, and AI processing, the script can analyze and categorize video content to identify trends, categorize scenes, and group videos by thematic tags or classification.

3) **Building a Searchable Video Library:**
   * Utilizes the script's ability to export processed data into JSON, integrating this JSON data into a MongoDB or Cosmos DB backend, allows for building a highly searchable video library complete with metadata and AI-derived descriptions.

# How It All Comes Together

Each module plays a vital role in achieving the overall functionality of the toolkit. `core.py` ensures the environment and essential resources (e.g., MongoDB connections) are correctly initialized. `workflow.py` orchestrates each step, from video download to final data export. Along the journey, `analysis.py` adds deep insights into the content, `tagging.py` organizes data into searchable categories, and `interface.py` provides an intuitive way to use the toolkit via the command line. Finally, `export.py` facilitates storing data in structured formats, ensuring the processed information is usable and accessible for further analysis, search, or machine learning purposes.

This structured approach leads to a flexible, extensible, and robust toolkit that can grow with new use cases as needed.

# Glossary of Functions: Purpose and Description

This glossary provides an in-depth description of the key functions and components within your video processing toolkit, explaining their purpose and how they contribute to the overall functionality.

## **1. `core.py`**
   * **`load_config(config_path='config.ini')`**:
	 * **Purpose**: Loads and parses the configuration settings from `config.ini`, providing access to configuration values for other modules.
	 * **Usage**: Called at the beginning of scripts or functions where configuration data (like API keys or file paths) is needed.
   * **`setup_logging(log_file='video_processing.log')`**:
	 * **Purpose**: Sets up the logging configuration, allowing the script to generate logs that track activity, errors, and performance.
	 * **Usage**: Initialized at the start of `main.py` or within other modules to set up logging compliance.
   * **`ensure_directory(directory)`**:
	 * **Purpose**: Checks if a specified directory exists; if it doesn't, the function will create it.
	 * **Usage**: Used prior to saving files or downloaded videos to ensure directories are in place.
   * **`connect_to_mongodb()`**:
	 * **Purpose**: Establishes a connection to a MongoDB database (or Cosmos DB for MongoDB), verifying accessibility by pinging the server.
	 * **Usage**: Called when database access is needed, particularly for storing or retrieving processed video data.
   * **`get_mongodb_collection()`**:
	 * **Purpose**: Retrieves the specific collection within a MongoDB database to interact with during the pipeline's operations.
	 * **Usage**: Used whenever the script needs to query, insert, or update database entries.
   * **`load_priority_keywords(file_path='priority_keywords.json')`**:
	 * **Purpose**: Loads a list of priority keywords from `priority_keywords.json`, which will be used to guide the AI's content analysis and description generation.
	 * **Usage**: Used in the `query_qwen_vl_chat` function to ensure the AI model prioritizes specific terms during analysis.
   * **`query_qwen_vl_chat(video_url, instruction="Describe this video.", use_vpc=False)`**:
	 * **Purpose**: Sends a request to the Qwen-VL-Chat model API over HTTPS, incorporating priority keywords if available, to generate a detailed description or analysis of a video.
	 * **Usage**: Integrated into the video processing pipeline to enrich metadata with AI-generated content analysis.
	 * **New Feature**: Now integrates with `priority_keywords.json` to modify the instruction/query sent to the AI model for focused content analysis.
   * **`is_video_url(url)`**:
	 * **Purpose**: Checks if a provided URL directly links to a video file based on its extension or content-type header.
	 * **Usage**: Used within the `download_video` function to determine if the URL is a direct video link or if further processing (such as scraping) is needed.
   * **`extract_video_url_from_html(html_content)`**:
	 * **Purpose**: Scrapes HTML content to find a likely video URL, searching for common elements like `<video>` or `<source>` tags or using regular expressions to match video URLs that may be embedded in JavaScript.
	 * **Usage**: Called within `download_video` when the URL does not directly point to a video file, aiding in the extraction of video links from webpages.

---

## **2. `analysis.py`**
   * **`extract_metadata(filepath)`**:
	 * **Purpose**: Extracts detailed metadata from a video file using FFmpeg, such as dimensions, FPS, codecs, etc., essential for further processing.
	 * **Usage**: Called early in the video processing pipeline, providing foundational data for analysis and tagging.
   * **`detect_scenes(filepath, threshold=0.3)`**:
	 * **Purpose**: Uses FFmpeg's scene detection to segment videos, storing important timestamps.
	 * **Usage**: Integrated during the video analysis phase to isolate segments of a video for targeted tagging.
   * **`process_with_qwen_vl(video_path, instruction="Describe this scene.")`**:
	 * **Purpose**: Processes a video by sending it to the Qwen2-VL model for detailed content analysis and returns relevant descriptions.
	 * **Usage**: Provides actionable insights into the video's content, aiding in tagging, classification, and metadata enrichment.

---

## **3. `tagging.py`**
   * **`load_custom_tags(custom_rules_path)`**:
	 * **Purpose**: Loads custom tagging rules from a JSON file specified by the user (`custom_tags.json`). This makes the tagging system extensible and adaptable to different needs.
	 * **Usage**: Called during the tagging phase to apply user-defined rules for tagging videos based on metadata, AI descriptions, or other criteria.
   * **`apply_tags(video_info, custom_rules=None)`**:
	 * **Purpose**: Applies a combination of default and custom tags to a video based on extracted metadata and AI-generated descriptions. Custom rules, if provided, from `custom_tags.json` are also used.
	 * **Usage**: Responsible for tagging videos according to existing and custom rules; the end-tags are crucial for video classification and search.
	 * **New Feature**: Integration with `custom_tags.json` to allow for more granular, user-defined tagging logic.
   * **`apply_custom_rules(video_info, tags, custom_tags_rules)`**:
	 * **Purpose**: Applies user-defined custom rules present in `custom_tags.json` to provide specific tags based on complex conditions.
	 * **Usage**: Enhances and refines tagging based on user-defined conditions or exceptions.
   * **`classify_video(video_info)`**:
	 * **Purpose**: Classifies a video into a high-level category (e.g., Action, Documentary, Cinematic) based on its tags and metadata.
	 * **Usage**: Called after tagging, assigning the video into predefined categories likely used in ML models or content libraries.

---

## **4. `interface.py`**
   * **`parse_arguments()`**:
	 * **Purpose**: Parses command-line input parameters, allowing users to customize things like output directories, API instructions, and more.
	 * **Usage**: Called by `main.py` to interpret user input and adjust the behavior of the script based on provided arguments.
   * **`main()`**:
	 * **Purpose**: The main entry point of the script, coordinating interactions between modules based on the user's command-line input.
	 * **Usage**: The first function executed when running the script, initializing necessary components and starting the video processing pipeline.

---

## **5. `workflow.py`**
   * **`process_video_pipeline(video_url, scene_threshold=None, custom_rules=None, output_dir=None)`**:
	 * **Purpose**: The centerpiece of the script, this function coordinates the complete video processing workflow for a single video, from downloading to metadata extraction, scene detection, tagging, classification, and finally, exporting processed data.
	 * **Usage**: This function drives the actual processing of videos by managing each step of the pipeline components and their respective dependencies.
   * **`download_video(url, filename)`**:
	 * **Purpose**: Downloads the video from the provided URL and stores it in a preset directory. Also checks if the URL is a direct link to a video and includes scraping logic to detect or extract video links if the URL points to an HTML page.
	 * **Usage**: Fundamental part of the pipeline's operation, this function handles the beginning stages by fetching the video content for processing.
	 * **New Feature**: Now includes logic to detect whether a URL is a direct video file; if it's an HTML page, it scrapes the page for video links and handles different head content types accordingly.

---

## **6. `export.py`**
   * **`export_to_json(data, output_dir)`**:
	 * **Purpose**: Handles the final step of the pipeline by exporting the processed video data, including metadata, tags, and classification, into a structured JSON file saved to a directory.
	 * **Usage**: Important for documenting and preserving the results of the video processing pipeline for further analysis, indexing, or use in machine learning tasks.

---

## **7. **Additional Features: Priority Keywords and Custom Tags**

### **`.json` Files & Their Purpose**

* **`priority_keywords.json`**:
   * **Purpose**: Provides a list of keywords or phrases the AI model should prioritize (or focus on) when analyzing and describing video content. This file is especially useful for steering the AI model towards specific terminology, enhancing the relevancy of its outputs.
   * **Integration**: Used within `core.py` in the `query_qwen_vl_chat` function to append these keywords to the AI's processing instructions.
   * **Usage**: Encourages the AI to emphasize certain themes or actions that are important for specific use cases, which may be less common or completely new.
* **`custom_tags.json`**:
   * **Purpose**: Defines a set of custom rules for tagging videos based on metadata, AI descriptions, and other video characteristics. It allows for applying specific tags that are not covered by the default logic, catering to specialized or uncommon criteria.
   * **Integration**: Loaded and parsed within `tagging.py`, specifically used in `apply_tags` and `apply_custom_rules` to augment the default tagging process.
   * **Usage**: Facilitates applying structured and detailed tags to videos based on user-defined standards, aiding in both manual curation and automated processes like content recommendation or classification.

### **Challenges & Strategies**

* **Handling Unfamiliar Keywords**:
   * When incorporating new or unfamiliar keywords from `priority_keywords.json` that the AI model might not fully recognize or interpret, prompt engineering and phrase association techniques can be used to guide the model's learning patterns passively.
   * Given the model's limitations, forcing it to learn entirely new terms would typically require model fine-tuning, which isn't feasible for all use cases. Instead, you can offer context for these terms in the `priority_keywords.json` to help the model infer their meanings through association.

---

# **Documentation: `custom_tags.json` And `priority_keywords.json`**

## Overview

`custom_tags.json` and `priority_keywords.json` are two JSON-based configuration files that extend the functionality of your video processing script. They allow for customized tagging and prioritization of keywords while interacting with AI models like Qwen2-VL for content analysis and description generation.

### **1. `custom_tags.json`**

`custom_tags.json` enables user-defined rules for tagging and classifying videos based on metadata, AI-generated descriptions, and other criteria. By leveraging this feature, you can automate the process of adding specific tags to videos according to tailored conditions that meet your project's requirements.

### **2. `priority_keywords.json`**

`priority_keywords.json` provides a list of keywords or phrases that the AI model should prioritize when analyzing and describing video content. This ensures that the generated content focuses on terms that are most relevant to your particular use case, emphasizing them in the analysis phase.

---

## **`custom_tags.json`**

### **Purpose**
* Custom tagging rules allow you to define how videos should be tagged based on criteria such as video metadata (e.g., duration, resolution) and the content descriptions generated by the AI model.
* Enables flexibility in adapting the tagging system to different domains, industries, or specific project requirements.

### **Structure Of `custom_tags.json`**

The file is expected to be a JSON object with an array of `rules`. Each rule can target the AI-generated description or video metadata and define tags to apply when specific conditions are met.

Here is an example structure of `custom_tags.json`:

```json
{
    "rules": [
        {
            "description_keywords": ["Aerial", "Drone", "Bird's Eye"],
            "tags": ["Aerial Shot", "Drone Footage"]
        },
        {
            "metadata_conditions": {
                "resolution": "4K",
                "duration_gt": 600
            },
            "tags": ["High Resolution", "Extended Play"]
        },
        {
            "ai_generated_tags": ["Action", "Explosive"],
            "tags": ["High-Intensity", "Action-Packed"]
        }
    ]
}
```

### **Explanation**
1) **`description_keywords`**:
   * Looks for specific keywords within the AI-generated description (result of the Qwen2-VL analysis).
   * If keywords like "Aerial," "Drone," or "Bird's Eye" are detected in the description, the tags "Aerial Shot" and "Drone Footage" are added to the video.

2) **`metadata_conditions`**:
   * Applies tags based on metadata, such as resolution and duration.
   * For instance, videos that are in 4K resolution and longer than 600 seconds get tagged with "High Resolution" and "Extended Play."

3) **`ai_generated_tags`**:
   * Applies tags based on the known AI-generated tags.
   * For instance, if the AI result includes tags like "Action" and "Explosive," the script will add "High-Intensity" and "Action-Packed."

### **Integration Into the Script**

The `custom_tags.json` functionality is integrated into the `tagging.py` file, and specifically within the `apply_tags()` and `apply_custom_rules()` functions.

* **`apply_tags(video_info, custom_rules=None)`**:
   * Loads the custom rules from `custom_tags.json` and applies them during tag creation. This ensures that user-defined rules are considered alongside default tagging behavior.
* **`apply_custom_rules(video_info, tags, custom_tags_rules)`**:
   * Called internally by `apply_tags()` to handle the application of the custom rules. It checks conditions defined in `custom_tags.json` and adjusts the tags accordingly.

### **Workflow Example**

When a video is processed:

1) Metadata is extracted and AI content analysis is performed.
2) The script checks the result against conditions defined in `custom_tags.json`.
3) Matching rules from `custom_tags.json` are applied, and corresponding tags are added to the video.
4) The video, along with its tags (both default and custom), is then classified and exported.

---

## **`priority_keywords.json`**

### **Purpose**
* The goal of `priority_keywords.json` is to provide guidance to the AI model, encouraging it to emphasize specific terms or themes during content analysis.
* Ensures that AI-generated descriptions contain key terms that are important for the task at hand, improving the relevance and applicability of the results.

### **Structure Of `priority_keywords.json`**

The file is simple and consists of a JSON object with an array of `priority_keywords`. These keywords or phrases influence how the AI model interprets and describes the video.

Here is an example structure of `priority_keywords.json`:

```json
{
  "priority_keywords": [
    "Aerial Shot",
    "Drone Footage",
    "Close-up",
    "Time-lapse",
    "Slow Motion",
    "Action Sequence",
    "Night Vision",
    "Documentary Footage",
    "Fast-paced",
    "Cinematic"
  ]
}
```

### **Explanation**
1) **`priority_keywords`**:
   * The array contains a list of keywords or phrases that you deem important for the specific context in which the video processing toolkit is being used.
   * When interacting with the AI model, these keywords are appended to the instruction/query guiding the AI to focus more on these terms when generating its content descriptions.

### **Integration Into the Script**

The `priority_keywords.json` functionality is integrated into the `core.py` file, particularly within the `query_qwen_vl_chat()` function.

* **`query_qwen_vl_chat(video_url, instruction="Describe this video.", use_vpc=False)`**:
   * This function sends the video URL and instruction to the Qwen-VL-Chat model API.
   * Before sending the request, it loads the priority keywords from `priority_keywords.json` and appends them to the instruction/query sent to the AI model.

## **Combining `custom_tags.json` And `priority_keywords.json`**

### **Scenario**

Imagine you're processing drone footage and want to ensure specific keywords like "Aerial Shot" and "Drone Footage" are prioritized in the AI's description. Simultaneously, you have specific custom rules in `custom_tags.json` that apply tags based on these keywords and other video metadata.

### **Workflow**
1) **Priority Keywords**: `priority_keywords.json` influences the AI model to emphasize "Aerial Shot" and "Drone Footage" during content analysis.
2) **AI Response**: If the AI detects these terms, the response will likely highlight them in its description.
3) **Custom Tagging**: When this description is evaluated, `custom_tags.json` rules look for these keywords and automatically apply relevant tags (e.g., "Aerial Shot").
4) **Output**: The final output includes a video tagged with "Aerial Shot" and categorized according to related criteria you've defined, all driven by priority keywords and custom tagging rules.

---

# Recap of File Structure & Functionality

* **`core.py`**: Handles configuration, essential utilities, logging, and the connection to MongoDB or Cosmos DB.
* **`analysis.py`**: Focuses on extracting and analyzing video metadata, scene detection, and obtaining AI-generated content from Qwen2-VL.
* **`tagging.py`**: Applies tags and classifications to videos based on extracted metadata and AI-driven insights.
* **`interface.py`**: Manages user input via command-line arguments, enabling customization of how the script.
* **workflow.py`**: The backbone the script, running the full processing pipeline from start to finish.
* **`export.py`**: Saves all processed data into a structured JSON format, making it retrievable for future analysis or as an input to a database.

This neatly organized file structure and the glossary of functions provide a thorough reference for understanding what each component does, and how they all work together to achieve the broader goals of your video processing toolkit.

---

# Complete & Updated Files

---

# **1. `core.py`**

This file contains the essential utility functions that the rest of the system relies on, as well as the function responsible for interacting with the Qwen2-VL-Chat API.

```python
import requests
import configparser
import logging
import os
import json
from logging.handlers import RotatingFileHandler

# Initial configurations and logging setup
config = configparser.ConfigParser()

def load_config(config_path='config.ini'):
    """Loads the configuration from a config file."""
    try:
        config.read(config_path)
        return config
    except configparser.Error as e:
        logging.error(f"Error reading config.ini: {e}")
        raise

def setup_logging(log_file='video_processing.log'):
    """Set up logging configuration."""
    log_handler = RotatingFileHandler(log_file, maxBytes=1024*1024, backupCount=5)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[log_handler])

def ensure_directory(directory):
    """Ensures that a directory exists."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        logging.info(f"Created directory: {directory}")
    else:
        logging.debug(f"Directory exists: {directory}")

def connect_to_mongodb():
    """Establishes a connection to MongoDB/CosmosDB."""
    try:
        from pymongo import MongoClient
        db_uri = config.get("MongoDB", "URI", fallback="your-default-uri")
        client = MongoClient(db_uri)
        # Since ping and connection checks might vary, a dedicated method is useful
        client.admin.command('ping')  
        return client
    except Exception as e:
        logging.error(f"MongoDB connection failed: {e}")
        return None

def get_mongodb_collection():
    """Retrieves the MongoDB collection after establishing a connection."""
    try:
        client = connect_to_mongodb()
        database_name = config.get("MongoDB", "DatabaseName", fallback="video_database")
        collection_name = config.get("MongoDB", "CollectionName", fallback="videos")
        db = client[database_name]
        collection = db[collection_name]
        return collection
    except Exception as e:
        logging.error(f"Could not retrieve MongoDB collection: {e}")
        return None

def load_priority_keywords(file_path='priority_keywords.json'):
    """Load priority keywords for analysis."""
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

## Explanation
* **Initialization**: The script starts by defining basic setup functions, including configuration loading and logging.
* **MongoDB Integration**: Functions are provided to connect to a MongoDB/Cosmos DB database and retrieve the appropriate collection.
* **Qwen-VL-Chat Interface**: The script includes flexibility to interact with both public and VPC endpoints for Qwen2-VL-Chat, using provided instructions and weaving in priority keywords.
* **Error Handling**: If connectivity issues or failed attempts are encountered when uploading to MongoDB or connecting to the Qwen2-VL-Chat API, detailed logging messages are generated.

---

# **2. `workflow.py`**

This file orchestrates the entire video processing pipeline, integrating the various functionalities such as downloading, analyzing, tagging, and exporting the processed data.

```python
import logging
from core import setup_logging, load_config, ensure_directory, query_qwen_vl_chat
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
    """Check if the URL directly links to a video file based on its extension or the headers."""
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.ogg']
    if any(url.endswith(ext) for ext in video_extensions):
        return True

    try:
        response = requests.head(url, allow_redirects=True)
        content_type = response.headers.get('Content-Type', '')
        if 'video' in content_type:
            return True
    except requests.RequestException as e:
        logging.error(f"Failed to check URL content type: {e}")

    return False

def extract_video_url_from_html(html_content):
    """Scrape HTML content to find a likely video URL."""
    soup = BeautifulSoup(html_content, 'html.parser')

    # Look for <video> tags
    video_tag = soup.find('video')
    if video_tag:
        video_url = video_tag.get('src')
        if video_url:
            return video_url

    # Look for <source> tags under <video> or standalone <source>
    source_tag = soup.find('source')
    if source_tag:
        video_url = source_tag.get('src')
        if video_url:
            return video_url

    # Look for URLs in embedded JS or other sources
    patterns = [
        r'(?:"|\')((http[s]?:\/\/.*?\.mp4|\.mkv|\.webm|\.ogg))(?:\"|\')',
        r'(?:"|\')((http[s]?:\/\/.*?video.*?))(?:\"|\')'
    ]

    for pattern in patterns:
        match = re.search(pattern, html_content)
        if match:
            return match.group(1)

    return None

def download_video(url, filename):
    """Downloads a video and stores it in the specified directory."""
    config = load_config()
    download_dir = config.get('Paths', 'DownloadDirectory', fallback='downloaded_videos')
    ensure_directory(download_dir)
    file_path = os.path.join(download_dir, filename)

    logging.info(f"Checking if URL {url} is a direct video link")

    if is_video_url(url):
        logging.info(f"URL is a direct video link, downloading from {url}")
        direct_url = url
    else:
        logging.info(f"URL {url} is not a direct video link, attempting to scrape the page for videos.")
        response = requests.get(url)
        if 'html' in response.headers.get('Content-Type', ''):
            direct_url = extract_video_url_from_html(response.text)
            if not direct_url:
                logging.error(f"Could not find a video link in the provided URL: {url}")
                return None
        else:
            logging.error(f"Content from URL {url} does not appear to be HTML.")
            return None

    try:
        response = requests.get(direct_url, stream=True)
        response.raise_for_status()

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
        return file_path
    except Exception as e:
        logging.error(f"Failed to download video: {e}", exc_info=True)
        return None

def process_video_pipeline(video_url, scene_threshold=None, custom_rules=None, output_dir=None):
    """Executes the full processing pipeline for a single video."""
    config = load_config()

    scene_threshold = scene_threshold if scene_threshold is not None else config.getfloat('SceneDetection',
                                                                                         'DefaultThreshold',
                                                                                         fallback=0.3)
    output_dir = output_dir if output_dir is not None else config.get('Paths', 'ProcessedDirectory', fallback='processed_videos/')
    
    logging.info(f"Starting pipeline for video: {video_url}")
    ensure_directory(output_dir)

    # Create a filename based on the URL
    filename = os.path.basename(urlparse(video_url).path)

    # Download the Video
    local_filepath = download_video(video_url, filename)
    
    if local_filepath:
        # Extract Metadata
        metadata = extract_metadata(local_filepath)
        if metadata:
            # Perform Scene Detection with the specified threshold
            scenes = detect_scenes(local_filepath, threshold=scene_threshold)

            # Process video using Qwen2-VL Chat Model via API
            instruction = config.get('QwenVL', 'DefaultInstruction', fallback="Describe this video.")
            qwen_description = query_qwen_vl_chat(video_url, instruction=instruction)

            # Aggregate metadata, scenes, and AI descriptions
            video_info = {**metadata, 'qwen_description': qwen_description, 'scenes': scenes}

            # Apply Tags and Classification
            tags = apply_tags(video_info, custom_rules=custom_rules)
            classification = classify_video(video_info)

            # Update video information with tags and classification
            video_info.update({
                'tags': tags,
                'classification': classification
            })

            logging.info(f"Video classified as {classification} with tags: {tags}")
            
            # Export the processed data to JSON in the specified output directory
            export_to_json(video_info, output_dir=output_dir)
        else:
            logging.error(f"Metadata extraction failed for {local_filepath}.")
    else:
        logging.error(f"Failed to download video from URL: {video_url}")
```

## Explanation
* **Video Download:** The `download_video` function has been enhanced to detect if a URL links directly to a video or if it needs to be scraped from an HTML page. This enables the pipeline to handle different types of video sources more effectively.
* **Pipeline Overview:** The `process_video_pipeline` coordinates different stages of video processing, such as downloading, metadata extraction, scene detection, AI interaction via Qwen-VL, tagging, classification, and exporting the final output.

---

# **3. `analysis.py`**

This file contains the necessary functions for analyzing video content, such as extracting metadata and detecting scene changes.

```python
import subprocess
import logging
import json
import os

def extract_metadata(filepath):
    """
    Extract video metadata using FFmpeg.

    Args:
        filepath (str): The path to the video file.

    Returns:
        dict: Extracted metadata like resolution, frame rate, codec, etc.
    """
    command = [
        'ffmpeg',
        '-i', filepath,
        '-v', 'quiet',
        '-print_format', 'json',
        '-show_format',
        '-show_streams'
    ]
    try:
        output = subprocess.check_output(command)
        metadata = json.loads(output.decode('utf-8'))
        logging.info(f"Extracted metadata for {filepath}")
        return {
            'duration': float(metadata['format']['duration']),
            'bit_rate': int(metadata['format']['bit_rate']),
            'resolution': f"{metadata['streams'][0]['width']}x{metadata['streams'][0]['height']}",
            'frame_rate': metadata['streams'][0]['r_frame_rate'],
            'codec': metadata['streams'][0]['codec_name'],
            'size': os.path.getsize(filepath)
        }
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to extract metadata: {e}")
        return None

def detect_scenes(filepath, threshold=0.3):
    """
    Detect scene changes in a video file using FFmpeg's `select` filter.

    Args:
        filepath (str): The path to the video file.
        threshold (float): Threshold for detecting a scene change.
        
    Returns:
        list: A list of timestamps where scene changes occur.
    """
    command = [
        'ffmpeg',
        '-i', filepath,
        '-vf', f'select=\'gt(scene,{threshold})\',metadata=print:file=-',
        '-an',
        '-f', 'null',
        '-'
    ]
    
    scene_timestamps = []
    try:
        output = subprocess.check_output(command, stderr=subprocess.STDOUT)
        for line in output.decode('utf-8').split('\n'):
            if 'pts_time:' in line:
                time_str = line.split('pts_time:')[1].split(' ')[0]
                scene_timestamps.append(float(time_str))
        logging.info(f"Detected {len(scene_timestamps)} scenes in {filepath}")
        return scene_timestamps
    except subprocess.CalledProcessError as e:
        logging.error(f"Scene detection failed: {e}")
        return []

```

## Explanation
* **`extract_metadata(filepath):`** This function extracts detailed metadata from a video file using FFmpeg. The extracted metadata includes important information such as duration, bitrate, resolution, frame rate, codec, and the file size. This metadata is returned as a dictionary for further processing in the pipeline.
* **`detect_scenes(filepath, threshold=0.3):`** This function uses FFmpeg's filter to detect scene changes in the video based on a specified threshold. The function returns a list of timestamps where scene changes occur, which can be used for more granular analysis or segmentation of the video.

---

# **4. `tagging.py`**

This file is responsible for applying tags and classifying videos based on extracted metadata, AI-generated descriptions, and custom rules.

```python
import logging
import json

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

## Explanation
* **`load_custom_tags(custom_rules_path):`** This utility function loads custom tagging rules from a specified JSON file (`custom_tags.json`). These custom rules allow the user to define how videos should be tagged based on metadata or AI-generated descriptions.
* **`apply_tags(video_info, custom_rules=None):`** This function applies tags to a video based on its metadata and AI-generated descriptions. It uses both default rules and, if provided, custom rules loaded from the JSON file. The tags are returned as a list and are free from duplicates.
* **`apply_custom_rules(video_info, tags, custom_tags_rules):`** It processes and applies the custom rules defined in the JSON file for tagging the video. It considers both video metadata and the AI-generated descriptions.
* **`classify_video(video_info):`** This function takes the tags applied to a video and returns a high-level classification ("Action", "Documentary", etc.). It can be expanded to include more categories or specificities as needed.

---

# **5. `export.py`**

This file is responsible for exporting the processed video data (including metadata, tags, and AI-generated descriptions) into a structured JSON format.

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

## Explanation
* **`export_to_json(data, output_dir):`** This function is responsible for exporting processed video data into a structured JSON format. It ensures that the output directory exists (creates it if necessary) and then writes the video data (including metadata, tags, and AI-generated descriptions) into a JSON file. The exported file is named after the video's filename and stored in the specified output directory with pretty-print formatting for easier reading.

---

# **6. `interface.py`**

This file handles command-line arguments and user inputs, allowing users to customize processing parameters dynamically.

```python
import argparse
import logging
from core import load_config, ensure_directory, setup_logging
from workflow import process_video_pipeline

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
    
    parser.add_argument('--urls', nargs='+', help='List of video URLs to process', required=True)
    parser.add_argument('--config', type=str, help='Path to the configuration file', default='config.ini')
    parser.add_argument('--output', type=str, help='Output directory', default=default_output)
    parser.add_argument('--log_level', type=str, help='Set the logging level',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], default='INFO')
    
    # Processing options
    parser.add_argument('--scene_threshold', type=float, help='Threshold for scene detection', default=default_scene_threshold)
    parser.add_argument('--qwen_instruction', type=str, help='Instruction passed to Qwen-VL', default=default_qwen_instruction)
    parser.add_argument('--use_vpc', action='store_true', help='Use the VPC endpoint for Qwen-VL API calls')

    args = parser.parse_args()
    return args

def main():
    """Main function that initializes the CLI and coordinates the processing."""
    args = parse_arguments()

    # Setup logging with the specified log level
    setup_logging(log_file='video_processing.log')
    logging.getLogger().setLevel(args.log_level)  # Set to the specific log level provided in the CLI arguments

    # Load configuration from the specified config file
    config = load_config(args.config)

    # Ensure output directory exists
    ensure_directory(args.output)

    # Iterate over provided URLs and process each video
    for url in args.urls:
        logging.info(f"Processing video from URL: {url}")
        
        # Run the full processing pipeline for the video
        process_video_pipeline(
            video_url=url,
            scene_threshold=args.scene_threshold,
            custom_rules=None,  # Optional: load custom rules if needed
            output_dir=args.output
        )

if __name__ == '__main__':
    main()
```

## Explanation
* **`parse_arguments():`** This function parses command-line arguments provided by the user. It allows users to specify numerous processing parameters, including video URLs for processing, output directories, log levels, scene detection thresholds, custom instructions for the Qwen2-VL model, and whether to use the public or VPC endpoint for API calls.
* **`main():`** The main function, which initializes the command-line interface, manages logging, and coordinates the video processing based on command-line inputs. It loads configuration settings, ensures directories exist, and iterates over the provided URLs to process each video using the `process_video_pipeline` function.

---

# **`config.ini`**

This is the configuration file where you define paths, thresholds, and other settings used by the script, including the Qwen-VL-Chat API endpoints and access keys.

```ini
[Paths]
# Directory where downloaded videos will be stored
DownloadDirectory = downloaded_videos

# Directory where processed videos and metadata will be exported
ProcessedDirectory = processed_videos

[MongoDB]
# Connection URI for MongoDB or Cosmos DB (MongoDB API)
URI = mongodb+srv://username:password@cluster.mongodb.net/video_database?retryWrites=true&w=majority
DatabaseName = video_database
CollectionName = videos

[Concurrency]
# Maximum number of workers (threads/processes) to use for parallel processing
MaxWorkers = 5

[QwenVL]
# Public endpoint for accessing the Qwen2-VL-Chat service
PublicEndpoint = https://quickstart-endpoint.region.aliyuncs.com/api/v1/inference

# VPC endpoint for accessing the Qwen2-Vl-Chat service (use for deployments within the same VPC)
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

## Explanation
* **Paths**: Paths for storing downloaded videos (`DownloadDirectory`) and processed data (`ProcessedDirectory`).
* **MongoDB**: Settings for connecting to your MongoDB or Cosmos DB instance, including the URI and database/collection names.
* **Concurrency**: Configuring the maximum number of workers for parallel processing.
* **QwenVL**: Configuration for the Qwen2-VL-Chat API, including public and VPC endpoints, and the access key.
* **SceneDetection**: The threshold used for detecting scene changes in the videos.
* **Rules**: The path to any custom tagging rules defined in `custom_tags.json`.
* **Logging**: The file where logs will be stored.

Replace values like `Your_Access_Key`, `URI`, etc., with actual values relevant to your deployment.

---

# **`requirements.txt`**

This is the file listing all the Python packages that need to be installed for the script to work. These packages include those for video processing, web requests, and other auxiliary tasks like logging and HTML scraping.

```txt
requests==2.28.1
configparser==5.2.0
beautifulsoup4==4.11.1
tqdm==4.64.1
pymongo==4.2.0
ffmpeg-python==0.2.0
lxml==4.9.1
mongoengine==0.26.0
```

## Explanation
* **requests**: Handles HTTP requests to download videos and interact with APIs.
* **configparser**: Parses the `config.ini` file.
* **beautifulsoup4**: Scrapes HTML content to find embedded video URLs.
* **tqdm**: Provides progress bars for the video download process.
* **pymongo**: Interacts with MongoDB or Cosmos DB using the MongoDB API.
* **ffmpeg-python**: Acts as a Python wrapper for FFmpeg, necessary for video processing tasks.
* **lxml**: Optional, enhances BeautifulSoup for faster and more reliable HTML/XML parsing.
* **mongoengine**: Object-Document Mapper (ODM) for MongoDB, used in more advanced MongoDB operations.

To install all dependencies, you can simply run:

```bash
pip install -r requirements.txt
```

---

# **`custom_tags.json`**

This file defines custom tagging rules. These rules allow you to specify specific tags that should be applied to videos based on certain metadata conditions or keywords found in AI-generated descriptions.

```json
{
    "rules": [
        {
            "description_keywords": ["Aerial", "Drone", "Bird's Eye"],
            "tags": ["Aerial Shot", "Drone Footage"]
        },
        {
            "metadata_conditions": {
                "resolution": "4K",
                "duration_gt": 600
            },
            "tags": ["High Resolution", "Extended Play"]
        },
        {
            "description_keywords": ["Action", "Explosive"],
            "tags": ["High-Intensity", "Action-Packed"]
        },
        {
            "metadata_conditions": {
                "codec": "h264",
                "bit_rate_gt": 2000000
            },
            "tags": ["High Bitrate", "H.264 Codec"]
        }
    ]
}
```

## Explanation
* **description_keywords**: These are keywords or phrases that are searched for within the AI-generated description of the video. If found, the specified tags are applied.
  * Example: Any video description containing "Aerial" or "Drone" will be tagged with "Aerial Shot" and "Drone Footage".
* **metadata_conditions**: These are conditions applied based on video metadata.
  * Example: A video with a 4K resolution and duration greater than 600 seconds will be tagged with "High Resolution" and "Extended Play".

You can expand this file with more rules to accommodate other specific tagging requirements.

---

# **`priority_keywords.json`**

This file lists the keywords you want the AI model (Qwen2-VL) to focus on when analyzing video content. These keywords help guide the AI to prioritize certain themes or terms in its descriptions.

```json
{
  "priority_keywords": [
    "Aerial Shot",
    "Drone Footage",
    "Close-up",
    "Time-lapse",
    "Slow Motion",
    "Action Sequence",
    "Night Vision",
    "Documentary Footage",
    "Fast-paced",
    "Cinematic"
  ]
}
```

## Explanation
* **priority_keywords**: These keywords influence how the AI model interprets and describes the video content. This could be useful to ensure that specific types of shots, themes, or content characteristics are emphasized in the AI's analysis.
  * Example: By including "Time-lapse" as a priority keyword, the AI is more likely to mention and emphasize time-lapse sequences in any video it processes, provided they are present.

You can customize this list according to the types of video content most relevant to your use case.

---

Based on the files and functionalities we've discussed, you should now have all the core components required to execute your video processing script. However, it's important to ensure that you've covered these aspects:

### **1. Script Components**
* **Core Script Files:**
  * `core.py`
  * `workflow.py`
  * `analysis.py`
  * `tagging.py`
  * `export.py`
  * `interface.py`
* **Configuration and Supporting Files:**
  * `config.ini`: Configuration settings, including API endpoints, paths, and threshold values.
  * `requirements.txt`: Dependencies list for Python packages required to run the toolkit.
  * `custom_tags.json`: Custom rules for tagging videos based on metadata and AI-generated content.
  * `priority_keywords.json`: Keywords you want the AI model to prioritize during content analysis.

### **2. Additional Considerations**
* **FFmpeg Installation:**
  The script relies on FFmpeg for video processing tasks like metadata extraction and scene detection. Ensure that FFmpeg is installed and accessible from your system's PATH. You can verify the installation with:

  ```bash
  ffmpeg -version
  ```

  If FFmpeg is not installed, you can download it from the [FFmpeg official website](https://ffmpeg.org/download.html) and follow the installation instructions for your operating system.

* **MongoDB/Cosmos DB Setup:**
  If you intend to use MongoDB for storing processed video data, ensure that:
  * You have a running instance of MongoDB or Cosmos DB with MongoDB API support.
  * The MongoDB URI is correctly configured in `config.ini`.
  * Necessary MongoDB collections are set up or will be automatically created when the script first connects.
* **Environment Variables (Optional):**
  If your setup requires sensitive information (like API keys) to be stored securely, consider using environment variables to avoid hardcoding credentials in `config.ini`. You can access these values in your script using:

  ```python
  import os
  access_key = os.getenv('QWEN_VL_ACCESS_KEY')
  ```

* **Python Version:**
  Ensure you are running a compatible Python version (typically 3.6 or higher) that supports all the libraries listed in your `requirements.txt`.

### **3. Additional Tools/Utilities**
* **Python Virtual Environment (Optional but Recommended):**
  Consider using a Python virtual environment to manage dependencies and avoid potential conflicts with other projects.

  ```bash
  python -m venv venv
  source venv/bin/activate  # On Windows: venv\Scripts\activate
  pip install -r requirements.txt
  ```

* **Logging and Monitoring (Optional):**
  Depending on your project's scale, you might want to consider enhancing logging. For instance:
  * **Alerting on Failures:** If you are running this as a long-running process or as part of a production pipeline, setting up email alerts or integrating with logging systems like ELK Stack could be useful.
  * **Cleanup Mechanism:** Depending on the volumes of video being processed, you may need mechanisms for cleanup or archiving old data/logs to prevent storage bottlenecks.

### **4. Execution**

Once all components are in place, you can execute your script via the command line. For example:

```bash
python interface.py --urls http://example.com/video.mp4 --output processed_videos --log_level INFO
```

If you have everything set up, running the command should trigger the entire video processing pipeline from downloading the video to exporting the analyzed and tagged data as JSON.

---

### **Project Roadmap: Enhancements, Extensions, and Streamlining**

This roadmap integrates the original ideas presented in the note with additional enhancements to create a comprehensive plan for the video processing toolkit's growth and scalability. The plan includes short-term manageable improvements, strategic long-term developments, and streamlining efforts for better user experience and system reliability.

---

### **1. Phase 1: Immediate Enhancements (1-2 Weeks)**
*Goal:* Strengthen the current infrastructure by improving performance, error handling, and user interface elements.

1) **Error Handling and Resilience**
   * **Automated Retry Logic:** Implement retry mechanisms for video downloads, scene detection, and API interactions to handle intermittent network failures.
   * **Graceful Failures:** Enhance error handling to ensure that failed tasks don't interrupt the entire pipeline.
   * **Failover for MongoDB:** Add MongoDB replica set support for high availability.

2) **Batch Processing with Queue System**
   * **Redis or Celery Integration:** Implement a queue system using Redis or Celery for processing multiple videos in parallel. This will utilize multiprocessing or threading to accelerate large-scale operations.
   * **Job Prioritization:** Integrate job priority settings within the queue system, allowing for the prioritization of urgent processing tasks.

3) **Unit and Integration Testing Framework**
   * **Unit Tests for Core Functions:** Implement unit tests for core functions using frameworks like `unittest` or `pytest`.
   * **CI Pipeline Integration:** Set up a continuous integration pipeline with GitHub Actions for automated testing.

4) **Thumbnail Generation at Key Points**
   * **FFmpeg Integration:** Use FFmpeg to generate thumbnails at scene changes or regular intervals.
   * **Scene Detection Enhancement:** Fine-tune the scene detection algorithm to generate more accurate keyframes.
   * **User Customization:** Allow users to configure thumbnail intervals or key points via configuration files.

5) **Enhanced Documentation**
   * **API Documentation:** Create detailed documentation for API endpoints using tools like Swagger.
   * **Expanded README:** Include setup instructions, dependencies (e.g., FFmpeg), and usage examples in the README.

6) **User Interface Enhancements**
   * **Enhanced CLI Features:** Upgrade the CLI to support interactive mode and batch processing from configuration files.
   * **Progress Monitoring:** Implement verbose logging and progress tracking options to give users better visibility into long-running processes.

---

### **2. Phase 2: Short-Term Extensions (3-4 Weeks)**
*Goal:* Extend the toolkit with advanced features and improve its performance and scalability.

1) **Integration with Cloud Storage**
   * **AWS S3, Google Cloud Storage, Azure Blob Support:** Implement the ability to download videos directly from and upload processed data to cloud storage services.
   * **Flexible Storage Options:** Allow storing metadata and outputs in cloud storage solutions alongside MongoDB, supporting multi-cloud architectures.

2) **Advanced Metadata Analysis and AI Integration**
   * **Semantic Video Tagging:** Utilize AI models like BERT for richer, context-aware tagging of video content.
   * **Optical Character Recognition (OCR):** Integrate Tesseract or similar libraries to extract text from video frames, enhancing content indexation.
   * **Object Detection and Classification:** Incorporate models like YOLO for object detection within videos, adding granular tagging based on recognized objects.

3) **Database Enhancements**
   * **Indexing and Query Optimization:** Optimize MongoDB index strategies, enabling faster read/write operations.
   * **Alternative Storage Backends:** Provide support for SQL databases like MySQL, PostgreSQL, and other NoSQL options, beyond MongoDB.

4) **Scalability and Performance**
   * **Batch Processing on Distributed Systems:** Enhance the script to support distributed computing frameworks like Apache Spark for large-scale processing.
   * **Parallel Processing:** Implement multi-threading/multi-processing for simultaneous video processing, significantly reducing time for large datasets.

5) **Basic Content Moderation**
   * **Computer Vision Techniques:** Introduce content moderation by detecting inappropriate or sensitive content using computer vision models.
   * **Automation:** Flag potentially harmful content for review, or automatically add warning tags.

---

### **3. Phase 3: Long-Term Vision (1-3 Months)**
*Goal:* Transform the toolkit into a robust, scalable, and extensible platform, accommodating advanced use cases and ensuring broader applicability.

1) **Customizable Plugins/Extensions Ecosystem**
   * **Plugin Architecture:** Establish a plugin system that allows users to write and integrate custom processing modules and AI models.
   * **Marketplace or Repository:** Create a central repository where plugins, workflows, and configuration templates can be shared and downloaded by the community.

2) **AI Model Training and Customization**
   * **Configurable AI Model Selection:** Allow users to select different AI models for content analysis, ensuring flexibility for various use cases.
   * **Active Learning Loop:** Implement a feedback mechanism for AI-generated tags or results, improving accuracy over time.

3) **Interactive Web Application**
   * **Web-Based UI:** Develop a user-friendly web interface, wrapping around the toolkit for real-time configuration, job submission, and monitoring.
   * **Searchable Video Library Interface:** Build an interface that displays all processed videos in an indexed and searchable format, enhancing data access and retrieval.

4) **API Endpoint Development**
   * **RESTful API Creation:** Offer a robust API to allow other services to interact with the toolkit programmatically.
   * **API Security and Throttling:** Implement security measures such as API key management and rate limiting.

5) **Video Summarization & Keyword Extraction**
   * **Summarization:** Deploy AI models to create concise text or video summaries of longer content.
   * **Keyword Generation:** Implement automatic keyword extraction to enhance searchability and categorization of video content.

6) **Cloud-Native and Multi-Platform Support**
   * **Dockerization:** Package the toolkit with Docker for easy deployment across various platforms.
   * **Serverless Architecture:** Build out serverless functions for lightweight processing and scalability.

7) **Progress Tracking and Visualization**
   * **Comprehensive Visualization Dashboard:** Create a real-time monitoring dashboard to track processing progress and visualize data insights.
   * **Logging and Auditing Tools:** Provide tools for detailed reports and logs of processing activities, necessary for auditing and compliance purposes.

---

### **4. Phase 4: Community and Ecosystem Building (Ongoing)**
*Goal:* Establish a strong, open-source community around the project, encouraging ongoing collaboration, experimentation, and innovation.

1) **Open Source Release**
   * **GitHub Repository:** Launch the project as open-source under a permissive license (e.g., MIT, Apache 2.0) and provide detailed contribution guidelines.
   * **Community Engagement:** Set up discussion forums (e.g., Discord, Gitter) to facilitate communications and support among users and contributors.

2) **Workshops and Tutorials**
   * **Online Courses:** Create educational materials, including video tutorials and live coding sessions, to help onboard users and contribute to the project.
   * **Hackathons:** Organize hackathons where developers and data scientists can collaborate on building new features or optimizing existing ones.

3) **Compliance Features**
   * **Regulation Compliance:** Add features designed to help users comply with relevant regulations (e.g., GDPR, CCPA), ensuring ethical and legal use of the toolkit.
   * **Data Anonymization:** Implement tools for anonymizing sensitive data within video content before processing or storage.

4) **Extensibility and Internationalization**
   * **Language Support:** Extend support for processing and analyzing videos in multiple languages, making the toolkit globally applicable.
   * **Plugin System for Custom Integrations:** Develop a plugin system enabling users to add or replace processing modules, such as for language localization or custom metadata extraction techniques.

---

### **Conclusion & Next Steps**

This roadmap integrates the original ideas with additional enhancements, offering a clear path to developing a robust, scalable, and flexible video processing toolkit. It addresses immediate needs, expands the toolkit's capabilities, and prepares it for adoption in diverse and large-scale applications.

Starting with the **Immediate Enhancements** in Phase 1, you'll build a solid foundation for more advanced features and widespread adoption, eventually fostering a community-driven ecosystem through open-source contributions and specialized plugins.

Would you like to focus on a specific phase first, or discuss the priority and feasibility of these tasks?

---

Okay, I understand you want to update your video processing toolkit repository to integrate the local deployment and inference of the Qwen2-VL model. 

Here's a plan for how we can achieve this, along with the necessary code changes:

**1. Update `requirements.txt`:**

   Add the required dependencies for Qwen2-VL and MoviePy:

   ```
   # ... (Your existing dependencies)
   transformers @ git+https://github.com/huggingface/transformers@21fac7abba2a37fae86106f87fcf9974fd1e3830
   modelscope
   accelerate
   qwen-vl-utils
   moviepy
   ```

**2. Create `qwen_model.py`:**

   Create a new file named `qwen_model.py` to handle Qwen2-VL model loading and inference:

   ```python
   import os
   import logging
   from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
   from modelscope import snapshot_download
   from qwen_vl_utils import process_vision_info
   import torch

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

       def process_video(self, video_frames, metadata, instruction="Describe this video."):
           """Processes a video using the Qwen2-VL model."""
           messages = [
               {
                   "role": "user",
                   "content": [
                       {"type": "video", "video": video_frames, "fps": metadata.get('fps', 1)},
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
               return output_text[0]
           except Exception as e:
               logging.error(f"Error processing video with Qwen2-VL: {e}", exc_info=True)
               return None
   ```

**3. Update `workflow.py`:**

   - Import the `QwenVLModel` class.
   - Load the model in the `process_video_pipeline` function.
   - Call the model's `process_video` method to get the description.

   ```python
   import logging
   from core import setup_logging, load_config, ensure_directory
   from analysis import extract_metadata, detect_scenes
   from tagging import apply_tags, classify_video
   from export import export_to_json
   import os
   import requests
   from tqdm import tqdm
   from urllib.parse import urlparse
   from bs4 import BeautifulSoup
   import re
   from qwen_model import QwenVLModel  # Import the QwenVLModel class

   # ... (Your existing functions: is_video_url, extract_video_url_from_html, download_video)

   def process_video_pipeline(video_url, scene_threshold=None, custom_rules=None, output_dir=None):
       """Executes the full processing pipeline for a single video."""
       config = load_config()

       scene_threshold = scene_threshold if scene_threshold is not None else config.getfloat('SceneDetection',
                                                                                            'DefaultThreshold',
                                                                                            fallback=0.3)
       output_dir = output_dir if output_dir is not None else config.get('Paths', 'ProcessedDirectory',
                                                                         fallback='processed_videos/')

       logging.info(f"Starting pipeline for video: {video_url}")
       ensure_directory(output_dir)

       # Create a filename based on the URL
       filename = os.path.basename(urlparse(video_url).path)

       # Download the Video
       local_filepath = download_video(video_url, filename)

       if local_filepath:
           # Extract Metadata
           metadata = extract_metadata(local_filepath)
           if metadata:
               # Perform Scene Detection with the specified threshold
               scenes = detect_scenes(local_filepath, threshold=scene_threshold)

               # Load the Qwen2-VL model
               qwen_model = QwenVLModel()  # Create an instance of the model

               # Process video using Qwen2-VL
               qwen_description = qwen_model.process_video(local_filepath, metadata)

               # Aggregate metadata, scenes, and AI descriptions
               video_info = {**metadata, 'qwen_description': qwen_description, 'scenes': scenes}

               # Apply Tags and Classification
               tags = apply_tags(video_info, custom_rules=custom_rules)
               classification = classify_video(video_info)

               # Update video information with tags and classification
               video_info.update({
                   'tags': tags,
                   'classification': classification
               })

               logging.info(f"Video classified as {classification} with tags: {tags}")

               # Export the processed data to JSON in the specified output directory
               export_to_json(video_info, output_dir=output_dir)
           else:
               logging.error(f"Metadata extraction failed for {local_filepath}.")
       else:
           logging.error(f"Failed to download video from URL: {video_url}")
   ```

**4. Update `config.ini`:**

   Add a section for Qwen2-VL and specify the model directory:

   ```ini
   [Qwen]
   ModelDir = /path/to/your/qwen2-vl-model
   ```

**Explanation:**

- **Model Loading:** The `QwenVLModel` class in `qwen_model.py` handles loading the model from the specified directory.
- **Pipeline Integration:** The `process_video_pipeline` function in `workflow.py` creates an instance of `QwenVLModel` and uses it to process the video.
- **Configuration:** The `config.ini` file allows you to specify the model directory.

**To Run:**

1. **Clone Repository:** Clone your repository to your Azure ML compute instance.
2. **Install Dependencies:** Install the required packages using `pip install -r requirements.txt`.
3. **Download Model:** Ensure the Qwen2-VL model is downloaded to the directory specified in `config.ini`.
4. **Execute Script:** Run your main script (e.g., `python interface.py ...`).

This setup will load the Qwen2-VL model locally on your Azure ML VM and use it for video processing within your toolkit. 

You're right to point that out! The current `analysis.py` module **does not directly use the Qwen model**. It primarily uses FFmpeg for metadata extraction and scene detection. 

The Qwen2-VL model is intended to be used in the `workflow.py` module, where you have the placeholder function `process_with_movienet`. This is where you would integrate the Qwen model to analyze the video content and generate descriptions or other insights.

**Here's a possible workflow for how `analysis.py` and the Qwen model could interact:**

1. **`analysis.py` Extracts Information:**
   - `extract_metadata`: Extracts technical metadata like resolution, duration, etc.
   - `detect_scenes`: Identifies scene changes in the video.

2. **`workflow.py` Orchestrates the Process:**
   - Calls the functions from `analysis.py` to get metadata and scene information.
   - Loads the Qwen2-VL model.
   - **Passes the video file (or segments) and the extracted metadata to the Qwen model for analysis.**
   - Receives the Qwen model's output (e.g., descriptions, tags).
   - Combines the Qwen output with the metadata and scene information.
   - Stores the results in the database or exports them to a file.

**In essence, `analysis.py` provides the foundational information (metadata and scenes) that can be used as input to the Qwen model, along with the video itself, to generate more meaningful insights about the video content.**

**Example of how `workflow.py` could use Qwen2-VL:**

```python
# ... (In your process_video_pipeline function in workflow.py)

# Load the Qwen2-VL model
qwen_model = QwenVLModel()

# Process each scene with Qwen2-VL
for scene in scenes:
    start_time = scene['start_time']
    end_time = scene['end_time']
    # Extract the scene segment (you might need to use MoviePy or FFmpeg for this)
    scene_segment_path = extract_scene_segment(local_filepath, start_time, end_time) 

    # Analyze the scene segment with Qwen2-VL
    qwen_description = qwen_model.process_video(scene_segment_path, metadata)

    # ... (Store the description and other information in your database or output file)
```

**Key Points:**

- **`analysis.py` provides the building blocks for video analysis.**
- **`workflow.py` orchestrates the process and integrates the Qwen model.**
- **The Qwen model uses the video content and metadata from `analysis.py` to generate insights.**

I hope this clarifies the relationship between `analysis.py` and the Qwen2-VL model! 


Okay, here's how you can implement the process of using `analysis.py` to provide input to the Qwen2-VL model in your `workflow.py` script:

**1. Update `analysis.py`:**

   - Add a function to extract video segments based on scene start and end times. You can use MoviePy for this.

   ```python
   import subprocess
   import logging
   import csv
   import json
   import os
   from core import query_qwen_vl_chat, ensure_directory
   from scenedetect import VideoManager, SceneManager
   from scenedetect.detectors import ContentDetector
   from moviepy.editor import VideoFileClip

   # ... (Your existing functions: segment_video, extract_metadata, detect_scenes)

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

**2. Update `workflow.py`:**

   - Import the `extract_scene_segment` function from `analysis.py`.
   - Modify the `process_video_pipeline` function to process each scene with Qwen2-VL.

   ```python
   import logging
   from core import setup_logging, load_config, ensure_directory
   from analysis import extract_metadata, detect_scenes, extract_scene_segment  # Import new function
   from tagging import apply_tags, classify_video
   from export import export_to_json
   import os
   from qwen_model import QwenVLModel

   # ... (Your existing functions: is_video_url, extract_video_url_from_html, download_video)

   def process_video_pipeline(video_url, scene_threshold=None, custom_rules=None, output_dir=None):
       """Executes the full processing pipeline for a single video."""
       config = load_config()

       scene_threshold = scene_threshold if scene_threshold is not None else config.getfloat('SceneDetection',
                                                                                            'DefaultThreshold',
                                                                                            fallback=0.3)
       output_dir = output_dir if output_dir is not None else config.get('Paths', 'ProcessedDirectory',
                                                                         fallback='processed_videos/')

       logging.info(f"Starting pipeline for video: {video_url}")
       ensure_directory(output_dir)

       # Create a filename based on the URL
       filename = os.path.basename(urlparse(video_url).path)

       # Download the Video
       local_filepath = download_video(video_url, filename)

       if local_filepath:
           # Extract Metadata
           metadata = extract_metadata(local_filepath)
           if metadata:
               # Perform Scene Detection with the specified threshold
               scenes = detect_scenes(local_filepath, threshold=scene_threshold)

               # Load the Qwen2-VL model
               qwen_model = QwenVLModel()

               # Process each scene with Qwen2-VL
               scene_descriptions = []
               for i, scene in enumerate(scenes):
                   start_time = scene['start_time']
                   end_time = scene['end_time']
                   logging.info(f"Processing scene {i+1} from {start_time:.2f}s to {end_time:.2f}s")

                   # Extract the scene segment
                   segment_path = extract_scene_segment(local_filepath, start_time, end_time)

                   if segment_path:
                       # Analyze the scene segment with Qwen2-VL
                       qwen_description = qwen_model.process_video(segment_path, metadata)
                       scene_descriptions.append({
                           'start_time': start_time,
                           'end_time': end_time,
                           'description': qwen_description
                       })
                   else:
                       logging.warning(f"Failed to extract scene segment {i+1}.")

               # Aggregate metadata, scenes, and AI descriptions
               video_info = {**metadata, 'scenes': scene_descriptions, 'filename': filename}

               # Apply Tags and Classification
               tags = apply_tags(video_info, custom_rules=custom_rules)
               classification = classify_video(video_info)

               # Update video information with tags and classification
               video_info.update({
                   'tags': tags,
                   'classification': classification
               })

               logging.info(f"Video classified as {classification} with tags: {tags}")

               # Export the processed data to JSON in the specified output directory
               export_to_json(video_info, output_dir=output_dir)
           else:
               logging.error(f"Metadata extraction failed for {local_filepath}.")
       else:
           logging.error(f"Failed to download video from URL: {video_url}")
   ```

**Explanation:**

- **`extract_scene_segment`:** This function in `analysis.py` extracts a video segment based on the provided start and end times using MoviePy.
- **Scene Processing Loop:** The `process_video_pipeline` function in `workflow.py` now has a loop that iterates through the detected scenes.
- **Qwen2-VL Analysis per Scene:** For each scene, the script extracts the corresponding segment using `extract_scene_segment` and then analyzes it with the Qwen2-VL model.
- **Scene Descriptions:** The descriptions generated by Qwen2-VL for each scene are stored in a list called `scene_descriptions`.
- **Aggregated Data:** The `video_info` dictionary now includes the `scene_descriptions` along with the metadata.

**Key Points:**

- **Scene-Level Analysis:** The Qwen2-VL model is now used to analyze each scene individually, providing more granular insights.
- **Data Structure:** The output JSON will now include a list of scene descriptions, each with its start and end times.
- **Error Handling:** The script includes error handling for scene segment extraction.

This implementation provides a more detailed and structured approach to video analysis by leveraging scene detection and analyzing each scene with the Qwen2-VL model. 
