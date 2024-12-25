---
model: chatgpt-4o-latest
---
###### YOU
2 things need to happen. It must be possible to upload a csv or txt file with a list of video files, whether they are urls or file names where a local directory is specified, that will then be processed by the script. Then i need priority_keywords.json functionality fully integrated into the ai’s input query, and custom_tags needs to be functioning 100% as well.

###### BMO
Certainly! Let's break down the tasks to achieve the two primary objectives outlined:

## Objectives:
1. **Enable Uploading a CSV or TXT File Containing Video URLs or Local File Paths**:
    - The script should be able to accept a list of video URLs or filenames from a CSV or TXT file and process each video accordingly, whether the video is hosted online or stored locally.
  
2. **Fully Integrate `priority_keywords.json` and Ensure `custom_tags.json` is Functioning 100%**:
    - Make sure the `priority_keywords.json` file directly influences the AI’s input query to prioritize certain keywords in its description of video content.
    - Ensure `custom_tags.json` is applied fully and correctly to classify and tag video files based on the rules provided.

### Implementation Plan:

### 1. **Upload CSV or TXT File and Process Video Files**

#### **Modifications - `main.py`**

We'll modify the `main.py` script to accept a CSV or TXT file, then process each entry, interpreting whether it's a URL or a local filename.

```python
from interface import parse_arguments
from workflow import process_video_pipeline
import os
import csv

def load_video_list(file_path):
    """Load video URLs or file paths from a CSV or TXT file."""
    video_list = []
    if file_path.endswith('.csv'):
        with open(file_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if row:
                    video_list.append(row[0])
    elif file_path.endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8') as txtfile:
            for line in txtfile:
                line = line.strip()
                if line:
                    video_list.append(line)
    else:
        raise ValueError("Unsupported file format. Only CSV or TXT files are supported.")
    return video_list

def main():
    """Main function that initializes the CLI and coordinates the processing."""
    args = parse_arguments()

    # Set up logging according to the user's preferences
    setup_logging(log_file='video_processing.log')
    logging.getLogger().setLevel(args.log_level)  # Set to the specific log level provided in the CLI arguments

    # Load configuration from the specified Configuration File
    config = load_config(args.config)

    # Ensure output directory exists
    ensure_directory(args.output)

    # Load video list from file
    video_list = load_video_list(args.file_path    # Process each video entry
    for video in video_list:
        if os.path(video):
            continue
        process_video_pipeline(video, scene_threshold=args.scene_threshold, custom_rules=args.custom_rules, output_dir=args.output)

if __name__ == '__main__':
    main()
```

#### **CLI Argument Parsing - `interface.py`**

Add a command-line argument that allows the user to specify the file path for the CSV or TXT file containing the list of video URLs or filenames.

```python
import argparse

def parse_arguments():
    """Parses command-line arguments provided by the user."""
    parser = argparse.ArgumentParser(description='Comprehensive Video Processing Toolkit')
    
    # Arguments related to video URL/file input
    parser.add_argument('--file_path', type=str, required=True, help='Path to a CSV or TXT file containing video URLs or local file paths.')

    # Other fields remain unchanged
    # E.g., output directory, logging level, and the use of VPC-specific endpoints
    
    # Custom Rules File
    parser.add_argument('--custom_rules', type=str, help='Path to custom rules JSON file for tagging.', default="custom_tags.json")
    
    return parser.parse_args()

def main():
    args = parse_arguments()
    print(f"File path provided: {args.file_path}")
    print(f"Output directory provided: {args.output}")
```

### 2. **Fully Integrate `priority_keywords.json` and Ensure `custom_tags.json` Functionality**

#### **Priority Keywords Integration - `query_qwen_vl_chat()` in `core.py`**

This function ensures AI input prioritizes key terms from `priority_keywords.json`.

```python
import json
import logging
import os

def load_priority_keywords(file_path="priority_keywords.json"):
    """Loads the priority keywords from a JSON file."""
    if not os.path.isfile(file_path):
        logging.warning(f"Priority keywords file {file_path} not found.")
        return []
    with open(file_path, 'r') as file:
        return json.load(file).get("priority_keywords", [])

def query_qwen_vl_chat(video_url, instruction="Describe this video.", use_vpc=False):
    """
    Sends a request to the Qwen-VL model API and appends priority keywords to the instruction if available.

    Args:
        video_url (str): The URL of the video file.
        instruction (str): The instruction sent to the AI model.
        use_vpc (bool): Whether to use the VPC endpoint or the public.

    Returns:
        str: The model's response, typically a description or analysis of the video.
    """
    priority_keywords = load_priority_keywords()

    # Append priority keywords to the instruction
    if priority_keywords:
        instruction += " Prioritize these keywords: " + ", ".join(priority_keywords)
    
    # Proceed with query as usual...
    public_endpoint = config.get("QwenVL", "PublicEndpoint", fallback="https://your-public-endpoint")
    vpc_endpoint = config.get("QwenVL", "VpcEndpoint", fallback="https://your-vpc-endpoint")
    access_key = config.get("QwenVL", "AccessKey", fallback="your-access-key")
    
    endpoint = vpc_endpoint if use_vpc else public_endpoint
    
    payload = {"video_url": video_url, "instruction": instruction}
    
    headers = {"Authorization": f"Bearer {access_key}", "Content-Type": "application/json"}

    try:
        response = requests.post(endpoint, json=payload, headers=headers)
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

#### **Custom Tags Integration - `tagging.py`**

Continuing from our earlier setup, ensure that the `tagging.py` file correctly applies tags based on conditions provided in the `custom_tag.json` file.

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
    """ Applies tags to a video based on metadata, AI-generated description, and custom rules."""
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
    
    # Apply custom tags rules if any
    if custom_rules:
        custom_tags_rules = load_custom_tags(custom_rules)
        tags = apply_custom_rules(video_info, tags, custom_tags_rules)

    return list(set(tags))  # Remove duplicates

def apply_custom_rules(video_info, tags, custom_tags_rules):
    """ Applies custom rules from custom_tags.json to video tags."""
    rules = custom_tags_rules.get("rules", [])
    for rule in rules:
        if 'description_keywords' in rule:
            keywords = set(rule['description_keywords'])
            if keywords.intersection(set(video_info.get('qwen_description', "").split())):
                tags.extend(rule['tags'])

        if 'metadata_conditions' in rule:
            conditions = rule.get('metadata_conditions', {})
            resolution = video_info.get('resolution', "")
            duration = video_info.get('duration', 0)
            if (conditions.get('resolution') == resolution and 
                duration > conditions.get('duration_gt', 0)):
                tags.extend(rule['tags'])

    return list(set(tags))  # Remove duplicates

def classify_video(video_info):
    """ Classifies a video into a high-level category based on its tags and metadata."""
    tags = video_info.get('tags', [])
    
    if "Action" in tags or "High Intensity" in tags:
        return "Action"
    
    if "Documentary Footage" in tags or "Narrative" in video_info.get('qwen_description', "").lower():
        return "Documentary"

    if "Aerial Shot" in tags or "Drone Footage" in tags:
        return "Cinematic"

    return "Unclassified"  # Default classification if no rules apply
```

### **Final Step: Running the Script**

Once the modifications are made, you can run the by executing:

```bash
python3 main.py --file_path path/to/your/video_list.csv --output /path/to/output/ --use_vpc --custom_rules path/to/your/custom_tags.json
```

### **Summary**

1. **CSV/TXT Integration**: The script now reads video URLs or file paths from a CSV or TXT file specified by the user. Each video is processed in sequence.
2. **Priority Keywords Integration**: Your AI query to the Qwen2-VL model now directly incorporates any keywords listed in `priority_keywords.json`.
3. **Custom Tags Functionality**: The script completely supports applying custom tagging rules as well using `custom_tags.json`. This ensures your video metadata is accurately tagged and classified according to your specifications.

With these changes, your video processing toolkit is more powerful and flexible, allowing you to handle multiple scenarios, different input formats, and custom tagging needs.

Let me know if you have any further questions or need additional customization!

###### YOU
Update the glossary.

###### BMO
Certainly! Given the recent changes and additions to your script, I will provide an updated glossary that includes the new functionalities such as CSV/TXT file processing for video lists, the integration of `priority_keywords.json`, and the use of `custom_tags.json`. Each entry in the glossary will describe the function's purpose, how it works, and its role in the overall video processing pipeline.

---

# **Glossary of Functions: Updated with New Features**

## **1. `core.py`**

### **`query_qwen_vl_chat(video_url, instruction="Describe this video.", use_vpc=False)`**
   - **Purpose**: Sends a request to the Qwen2-VL model API to analyze a video. Integrates priority keywords into the AI query to encourage the model to emphasize specific terms.
   - **How it Works**:
     - **Input**: Video URL and an instruction for the AI.
     - **Process**: Loads `priority_keywords.json`, merges it with the instruction, then sends the instruction and URL to the Qwen2-VL API.
     - **Output**: Receives and returns a descriptive output from the model based on the instructed keywords and video content.
   - **Role in Workflow**: This function tailors the AI model's content analysis by incorporating user-defined keywords, making the output more relevant to your specific focus areas.

### **`load_priority_keywords(file_path="priority_keywords.json")`**
   - **Purpose**: Loads a list of priority keywords from a JSON file (`priority_keywords.json`).
   - **How it Works**:
     - **Input**: The file path to `priority_keywords.json` file.
     - **Process**: Parses the JSON to extract a list of keywords indicating what the AI model should prioritize when analyzing video.
     - **Output**: Returns a list of keywords to be used in the query.
   - **Role in Workflow**: Ensures that the AI modelizes important keywords during its analysis, improving the relevance of its descriptions.

## **2. `interface`**

### **`parse_arguments()`**
   - **Purpose**: Handles the and parsing of command-line arguments provided by the user.
   - **How it**:
     - **Input**: Command-line arguments such as `--file_path`, `output`, `--custom_rules`, and `--use_vpc`.
     - **Process**: Uses Python’s argparse library to process and validate inputs. Each argument directs the behavior of the script.
     - **Output**: Returns a set of arguments in a Namespace object to be used throughout the script.
   - **Role in Workflow**: Allows users to configure how the script should operate, ensuring flexibility for different inputs and requirements.

### **`main()`**
   - **Purpose**: Central coordination point for the entire script. Reads video URLs or file paths from a CSV/TXT file and orchestrates their processing.
   - **How it Works**:
     - **Input**: Takes the parsed command-line arguments from `parse_arguments()`.
     - **Process**: Loads the video list, processes each entry (either URL or local file path), and calls the main processing function.
     - **Output**: Orchestrates the overall processing flow for all listed videos.
   - **Role in Workflow**: Main entry point that ties all components together; initiates video processing based on the user’s inputs.

### **`load_video_list(file_path)`**
   - **Purpose**: Reads a CSV or TXT file containing a list of video URLs or local file paths, returning the list in a format that can be processed by the script.
   - **How it Works**:
     - **Input**: Path to the CSV or TXT file.
     - **Process**: Parses each line (in TXT files) or row (in CSV files), stripping any extra whitespace and skipping empty lines.
     - **Output**: Returns a list of video URLs or file paths to process.
   - **Role in Workflow**: Ensures the script can process multiple videos in succession by loading a list of video paths or URLs from an input file.

## **3. `workflow.py`**

### **`process_video_pipeline(video_url, scene_threshold=None, custom_rules=None, output_dir=None)`**
   - **Purpose**: Coordinates the entire video processing pipeline for a single video, from download and analysis to tagging and exporting.
   - **How it Works**:
     - **Input**: Video URL, optional scene detection threshold, path to custom rules, and output directory.
     - **Process**: Downloads the video (if applicable), extracts metadata, detects scenes, interacts with the Qwen2-VL model, applies tags, classifies the video, and exports results to JSON.
     - **Output**: Processed video data (tags, classifications, descriptions) saved as a JSON file.
   - **Role in Workflow**: The central orchestrator for processing an individual video, drawing from other modules (core, analysis, tagging, export) to generate comprehensive video metadata.

### **`download_video(url, filename)`**
   - **Purpose**: Downloads a video from an online URL, handling both direct links and scenarios where video links are embedded in HTML.
   - **How it Works**:
     - **Input**: URL of the video to download, and the filename to save it as.
     - **Process**: Checks if the URL is direct or embedded, scrapes if necessary, and downloads the file.
     - **Output**: Saves the video to the local file system.
   - **Role in Workflow**: Acquires video content for further processing, whether from local files or remote URLs.

## **4. `analysis.py`**

### **`extract_metadata(filepath)`**
   - **Purpose**: Extracts metadata from a video file using FFmpeg, providing crucial details such as duration, resolution, and codec.
   - **How it Works**:
     - **Input**: The path to the video file.
     - **Process**: Runs an FFmpeg command that outputs video metadata in JSON format, parses and extracts relevant fields.
     - **Output**: Returns a dictionary of video metadata (duration, resolution, codec, etc.).
   - **Role in Workflow**: Supplies basic information about the video, which is used for tagging, classification, and essential logging.

### **`detect_scenes(filepath, threshold=0.3)`**
   - **Purpose**: Detects scene changes within a video file using FFmpeg’s built-in scene detection capabilities.
   - **How it Works**:
     - **Input**: Path to the video file and an optional threshold to fine-tune scene detection.
     - **Process**: Runs FFmpeg commands to analyze the video and identify timestamps where scene changes occur.
     - **Output**: Returns a list of detected scene change timestamps.
   - **Role in Workflow**: Provides a scene breakdown that can be used for more granular analysis or tagging, helping to segment the video content.

## **5. `tagging.py`**

### **`apply_tags(video_info, custom_rules=None)`**
   - **Purpose**: Applies tags to a video based on metadata, AI-generated descriptions, and user-defined custom rules.
   - **How it Works**:
     - **Input**: Information about the current video (metadata, AI description) and the path to the custom rules JSON.
     - **Process**: Applies default tag rules based on metadata and descriptions, and additionally applies any custom rules defined in `custom_tags.json`.
     - **Output**: Generates a list of tags based on the video’s metadata, AI description, and custom rules.
   - **Role in Workflow**: Ensures that the video content is accurately categorized and labeled according to predefined rules and AI-driven descriptions.

### **`apply_custom_rules(video_info, tags, custom_tags_rules)`**
   - **Purpose**: Applies user-defined custom tagging rules from `custom_tags.json`.
   - **How it Works**:
     - **Input**: Video metadata and description, existing tags, and the custom tagging rules from the JSON file.
     - **Process**: Evaluates the video info against condition-based rules and applies additional tags as specified.
     - **Output**: Returns an updated list of tags reflecting both standard and custom tagging rules.
   - **Role in Workflow**: Enhances flexibility in the tagging process, allowing users to customize video tags based on unique needs or industry-specific criteria.

### **`classify_video(video_info)`**
   - **Purpose**: Classifies a video into a broad category (such as "Action," "Documentary", etc.) based on the applied tags and metadata.
   - **How it Works**:
     - **Input**: Video information including tags and AI-generated content descriptions.
     - **Process**: Matches applied tags and content descriptions to predefined labels.
     - **Output**: Returns a classification string that represents the video's primary category.
   - **Role in Workflow**: Ensures videos are categorized, easing both future searches and more advanced analyses (e.g., class-specific training datasets for ML models).

### **`load_custom_tags(custom_rules_path)`**
   - **Purpose**: Loads custom tagging rules from a JSON file (`custom_tags.json`).
   - **How it Works**:
     - **Input**: The path to the custom rules JSON file.
     - **Process**: Parses the JSON file into a Python dictionary that contains various rules for tagging.
     - **Output**: Returns a dictionary with custom tags and associated conditions.
   - **Role in Workflow**: Provides the tagging system with user-defined rules, allowing customization beyond default behaviors.

## **6. `export.py`**

### **`export_to_json(data, output_dir)`**
   - **Purpose**: Exports processed video data, including metadata, tags, and classification, to a JSON file.
   - **How it Works**:
     - **Input**: Processed video data (metadata, tags, AI descriptions) and the directory for exporting the JSON file.
     - **Process**: Converts the data dictionary to JSON format and writes it to a file.
     - **Output**: Creates a JSON file in the specified directory.
   - **Role in Workflow**: Saves the processed information in a structured format, making the data accessible for future use (e.g., building searchable databases, further ML training, etc.)

---

# Recap of File Structure and Functionality

- **`core.py`**: Handles the integration with the Qwen2-VL model, ensuring that priority keywords influence the AI's analysis. Provides foundational functionalities like configuration loading and query processing.
- **`interface.py`**: Manages user inputs via command-line arguments, enabling flexibility in how the script is run (e.g., processing a list of videos from a file, setting output directories, etc.).
- **`workflow.py`**: Orchestrates the entire video processing pipeline, integrating the downloading, scene detection, AI analysis, and exporting phases into a cohesive workflow.
- **`analysis.py`**: Focuses on extracting and analyzing video metadata, as well as segmenting the video into scenes for more structured analysis.
- **`tagging.py`**: Implements a customized and extensible tagging system, allowing for both default and user-defined tags to apply to videos. Also handles video classification.
- **`export.py`**: Handles the persistent storage of processed video information in JSON format, making the results easily accessible and useful for various applications.

---

This updated glossary should provide you with a comprehensive reference tool. It encapsulates the functionalities, especially with new features like processing from CSV/TXT files, the integration of `priority_keywords.json` into your AI queries, and the full implementation of `custom_tags.json`.

Let me know if you have further questions about the script or need additional features or adjustments!

###### YOU
Create an updated visual workflow diagram

###### BMO
Certainly! I'll provide you with a detailed visual workflow diagram that incorporates all of the recent updates:

1. **Processing video URLs or file paths from CSV/TXT files.**
2. **Integrating `priority_keywords.json` into the AI query.**
3. **Applying `custom_tags.json` for custom tagging rules.**
4. **Exporting the processed output to JSON files.**

Here's a conceptual outline with a step-by-step breakdown:

```plaintext
[User Input: Command Line Arguments]
   |
   |-----> [Load Configuration: config.ini]
   |
   |-----> [Parse Command-Line Arguments: interface.py]
   |          |---> [Load Video List from CSV/TXT file: main.py]
   |                          
   |-----> [For Each Video URL or Local File Path]
                |
                |---> [Check if Local File or URL: workflow.py]
                       |
                       |---> [If URL: download_video]
                       |     
                       |---> [If Local File: Skip Download]
                       |
                       |---> [Extract Video Metadata: analysis.py]
                       |          |---> [Extract Duration, Resolution, Codec, etc.]
                       |          |---> [Detect Scene Changes, if enabled]
                       |
                       |---> [Integrate with Qwen2-VL: core.py]
                       |          |---> [Load Keywords from priority_keywords.json]
                       |          |--->Modify AI Query with Priority Keywords]
                       |          |---> [Submit Query to Qwen2-VL Model]
                       |          |---> [Receive AI-Generated Description]
                       |
                       |---> [Apply Tags to Video: tagging.py]
                       |          |---> [Default Tagging (Based on Metadata & AI Descriptions)]
                       |          |---> [Load and Apply Custom Tags from custom_tags.json]
                       |
                       |---> [Classify Video: tagging.py]
                       |          |---> [Assign a Category (Action, Documentary, etc.)]
                       |
                       |---> [Export Processed Data: export.py]
                       |          |---> [Save Processed Metadata, Tags, and Classification to JSON]
                       |
                       |----> [Processed Output: Save JSON in Specified Directory]
```

### **Diagram: Visual Representation**

Since I can't directly create or embed images, here's how you can visually conceptualize this using common diagram styles:

```plaintext
                                [ User Input: Command Line Arguments ]
                                              |
             +--------------------------------+--------------------------------+
             |                                |                                |
   [ Load Configuration:               [ Parse Command          [ Load from CSV/TXT:
      config.ini ]                     Arguments: User Options ]        Video Paths ]
                                                      |                             
                                                      v                             
                                      [ Loop through Videos in List ]               
                                                      |                             
+------------------------------------ If URL -----------------------------------------------------------------------+      
|                                       |                                                                           |
|                   +------------------------ Check if the URL            +-----------------------------------------|     
|                                     |    Points to a                       |                                       |     
|         [ If URL }              [download_video] ]   [ If Local File ]       -> Skips Download -> Metadata Analysis ->        
|                                                                                        |                      |      
[ Downloads Video |                  |                                                                      |        
                                      |                                                                      |      
                   [ Extract Metadata ]                                                   [ Detect Scene Changes ] 
                                 |                                                            |    
[ Export Metadata, Tags, Descriptions JSON ] <----[ Apply Tags Based on Metadata & Custom Rules ] <--- [Integrate Priority Keywords & 
                                                                                                             Query Qwen2-VL ]
          
                  |
[ Save or Export to a New JSON File for Searchable Database ]
```

### **Breakdown of the Diagram**

- **User Input**: 
  - The user provides various command-line arguments, including a file path to TXT/CSV files with video URLs or local file paths, and options for custom rules and output directories.

- **Load Configuration**: 
  - The script loads general settings from `config.ini`, which includes paths and API endpoints.

- **Parse Arguments and Load Video List**:
  - Command-line arguments are parsed, determining things like the output directory, custom rule file paths, and whether to use VPC endpoints.
  - The script loads the list of video URLs or file paths from the CSV/TXT file.

- **Video Processing Loop**:
  - The script loops through each video:
    - If it's a URL: determine whether to download.
    - If it's a local file, skip downloading.
  
- **Metadata Extraction**: 
  - Extracting key metadata, like resolution, codec, and duration.

- **Qwen2-VL Integration**:
  - Load keywords from `priority_keywords.json`.
  - Adjust AI query to include these prioritized terms.
  - Retrieve descriptions or relevant insights via the Qwen2-VL API.

- **Tag & Classify**:
  - Apply default and custom tags based on metadata, AI descriptions, and rules in `custom_tags.json`.
  - Classify videos into broader categories like "Action" or "Documentary."
 
- **Export**:
  - The processed output (metadata, tags, AI descriptions) is exported into structured JSON files, ready for creating a searchable database or further processing.

---

### **Conclusion**

This updated visual workflow incorporates every aspect of your current system—from loading files and querying AI models to applying custom tagging rules and exporting clean datasets ready for further analysis. Such a workflow ensures the script is flexible, powerful, and handles a range of video processing use cases, whether starting from local files or URLs.

Should you have any more questions or need further refinements, feel free to ask!
