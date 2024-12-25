# ResNet-Based Metadata Tagging for Plex Media Library: An In-Depth Guide

This guide provides a comprehensive, step-by-step approach to implementing ResNet (Residual Networks) for metadata tagging within your Plex Media Library. The goal is to automate the process of analyzing video content, extracting relevant tags from keyframes, and updating your Plex library with this information.

By the end of this guide, you’ll have a fully functioning pipeline that extracts keyframes from videos, classifies them using ResNet, and updates your Plex library with the generated metadata.

## Table of Contents

1. **Prerequisites**
2. **Step 1: Environment Setup**
3. **Step 2: Frame Extraction from Videos**
4. **Step 3: Loading and Using a Pre-trained ResNet Model**
5. **Step 4: Classifying Keyframes with ResNet**
6. **Step 5: Aggregating Tags Across Video Segments**
7. **Step 6: Updating the Plex Library with New Metadata**
8. **Conclusion**

---

### 1. Prerequisites

Before we dive into the implementation, ensure you have:

- A working installation of Plex Media Server on your system.
- Python installed (preferably version >= 3.8).
- Basic understanding of Python programming.
- Access to a GPU is recommended for faster processing but not mandatory.
  
### Required Packages:

```bash
pip install opencv-python-headless tensorflow requests
```

---

### 2. Step 1: Environment Setup

First, let's set up our environment by installing all necessary libraries.

```bash
pip install opencv-python-headless tensorflow requests numpy
```

Here’s what each package does:

- `opencv-python-headless`: Used for video processing and frame extraction.
- `tensorflow`: Provides access to pre-trained models like ResNet.
- `requests`: Allows us to interact with the Plex API for updating metadata.
- `numpy`: Useful for handling arrays and image data.

### Directory Structure

Ensure your project's directory is organized as follows:

```
PlexResNet/
│
├── videos/                 # Directory containing your video files
├── frames/                 # Directory where extracted frames will be stored
├── metadata/               # Directory where generated metadata (tags) will be stored
└── main.py                 # Main script containing the implementation
```

Create these directories before proceeding.

---

### 3. Step 2: Frame Extraction from Videos

We begin by extracting keyframes from each video file in the `videos/` directory.

**Script: frame_extraction.py**

```python
import cv2
import os

def extract_frames(video_path, output_dir, interval=5):
    """
    Extracts frames from a video at given intervals.

    Args:
        video_path (str): Path to the input video file.
        output_dir (str): Directory where extracted frames will be saved.
        interval (int): Interval in seconds between extracted frames.

    Returns:
        None
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    current_frame = 0

    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        if current_frame % (interval * fps) == 0:
            frame_filename = os.path.join(output_dir, f"frame_{current_frame}.jpg")
            cv2.imwrite(frame_filename, frame)
        
        current_frame += 1
    
    cap.release()

def process_all_videos(input_dir='videos/', output_dir='frames/', interval=5):
    """
    Processes all videos in a directory and extracts frames at specified intervals.

    Args:
        input_dir (str): Directory containing input videos.
        output_dir (str): Parent directory where all frames will be saved.
        interval (int): Seconds between each frame extraction.

    Returns:
        None
    """
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for video_file in os.listdir(input_dir):
        if video_file.endswith((".mp4", ".avi", ".mkv")):
            video_path = os.path.join(input_dir, video_file)
            frame_output_path = os.path.join(output_dir, os.path.splitext(video_file)[0])
            
            extract_frames(video_path, frame_output_path, interval)

if __name__ == "__main__":
    process_all_videos()
```

**How It Works**:

- The script reads each video file in `videos/` and extracts frames at an interval you specify (`interval` parameter).
- Frames are saved in subdirectories within `frames/`, named after their respective videos.

Run this script:

```bash
python frame_extraction.py
```

This will populate your `frames/` directory with extracted keyframes ready for classification.

---

### 4. Step 3: Loading and Using a Pre-trained ResNet Model

Next up is loading a pre-trained ResNet model using TensorFlow's Keras API to classify these keyframes into relevant categories or tags.

**Script Snippet: load_resnet.py**

```python
import tensorflow as tf

def load_resnet_model():
    """
    Loads a pre-trained ResNet model using Keras applications module.
    
    Returns:
        model (tf.keras.Model): Loaded ResNet model ready for inference.
     """
    
   # Load pre-trained ResNet50 model + higher level layers 
   resnet_model = tf.keras.applications.ResNet50(weights='imagenet')

   return resnet_model
   
if __name__ == "__main__":
   model = load_resnet_model()
   print(model.summary())
```

**How It Works**:

- We load a pre-trained version of ResNet50 that has been trained on ImageNet—a large dataset containing millions of images across thousands of categories—to leverage transfer learning capabilities right out-of-the-box without needing custom training data right away!

Run this snippet standalone first to confirm that TensorFlow can properly load the model on your machine:

```bash
python load_resnet.py
```

You should see a summary of the loaded model structure printed out on screen if everything went fine!

---

### Step 5.: Classifying Keyframes With ResNet


Now it's time actually classify those previously extracted keyframe images into meaningful tags leveraging our freshly loaded fine-tuned res-net50 neural network architecture!


Script Snippet classify_keyframe .py : 

``` python 
import tensorflow as tf 
from keras.preprocessing import image 
from keras.applications.resnet50 import preprocess_input , decode_predictions 
import numpy as np 
import json 
import os 

def classify_keyframe( img_path , model ): 

"""
Classifies single image via provided RESNET deep convolutional network architecture . 

Args : img path str : File path pointing towards individual jpg/png/bmp format photograph needing classification . Model tf keras Model Loaded RESNET machine-learning architecture capable transforming raw pixel data towards categorical labels via inferencing stage .
Returns :
Dictionary structure encompassing both top predicted label alongside its associated probability percentage value obtained through inferencing stage .
"""

img image.load_img( img path , target size =(224 ,224 )) 
x image.img_to_array( img ) 
x np.expand_dims x axis -0 ) x preprocess_input x )
preds model.predict( x ) results decode_predictions preds top=1 )[0][0] return { 'label': results [1], 'probability': float(results [2]) } 

def batch_classify_images_in_folder(folder_path ,model ): 

"""
Batch processes every single jpg/png/bmp format photograph found residing inside target folder returning structured JSON formatted result per respective classified item it encounters sequentially along way .

Args : folder path str Path pointing towards folder location housing multiple jpg/png/bmp format photographs necessitating classification run via RESNET deep convolutional neural-network architecture .

Returns :
List structured format encompassing individual dictionary structures representing top predicted label alongside its associated probability percentage value obtained through inferencing stage per respective file processed .
"""

all results [] filenames sorted(os.listdir(folder path)) filename filenames : fullpath=os.join(folder path filename ) labels classify_keyframe(fullpath model) result { 'file':filename 'metadata': labels } all results.append result return all results 

if main : '' main py batch classify images folder ('./frames /video name',load res net model ()) print json.dumps(json.dump(all_results,' pretty ',' True '))
~ 
```
#### How it Works :

• This function first loads individual JPEG photograph(s) found residing inside chosen folder location designated earlier during previous stages followed immediately thereafter invoking same previously pretrained RESNET mechanism previously loaded earlier upon aforementioned function subsequently outputs resultant categorial predictions probabilities derived thereof accordingly then finally stores those predictions categorically organized JSON formatted structure finally returning overall collection back main calling sequence refactored later eventually! 

---
 
### 6. Step 6: Aggregating Tags Across Video Segments

Once we have classified all the frames in a video, we need to aggregate the tags to create a summary of the video content. This step will help ensure that the most relevant tags are extracted from multiple frames and used to represent the entire video.

**Script Snippet: `aggregate_tags.py`**

```python
import json
import os
from collections import Counter

def aggregate_tags_from_frames(frame_metadata_dir):
    """
    Aggregates tags from classified frames across a video.

    Args:
        frame_metadata_dir (str): Directory where metadata JSON files are stored for each frame.

    Returns:
        dict: Dictionary containing aggregated tags and their counts.
    """
    
    all_tags = []

    for json_file in os.listdir(frame_metadata_dir):
        if json_file.endswith(".json"):
            with open(os.path.join(frame_metadata_dir, json_file), 'r') as f:
                frame_data = json.load(f)
                tag = frame_data['metadata']['label']
                all_tags.append(tag)

    # Aggregate occurrences of each tag using Counter
    aggregated_tags = Counter(all_tags)

    return dict(aggregated_tags)

def save_aggregated_tags(video_name, aggregated_tags, output_dir='metadata/'):
    """
    Saves aggregated tags to a JSON file.

    Args:
        video_name (str): Name of the video file.
        aggregated_tags (dict): Dictionary containing aggregated tags and their counts.
        output_dir (str): Directory where the final metadata file will be saved.

    Returns:
        str: Path to the saved metadata file.
    """

    output_path = os.path.join(output_dir, f"{video_name}_metadata.json")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    metadata = {
        'video': video_name,
        'tags': aggregated_tags
    }

    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=4)

    return output_path

if __name__ == "__main__":
    
   # Example usage for one video's frames
   example_video_frames_folder = "./frames/video_name"
   
   # Aggregate tags from individual frames
   aggregated_results = aggregate_tags_from_frames(example_video_frames_folder)
   
   # Save these results into metadata directory 
   save_aggregated_tag_path = save_aggregated_tags("video_name", aggregated_results)
   
   print(f"Aggregated metadata saved at: {save_aggregated_tag_path}")
```

**How It Works**:

- The script reads through all JSON files generated from classifying keyframes and aggregates the most frequent labels using Python's `Counter`.
- This provides a summarized view of what objects or scenes dominate within that particular video segment.
- Finally, it stores this summary as a JSON file which can be used later when updating Plex library metadata.

Run this script after you’ve classified your keyframes:

```bash
python aggregate_tags.py
```

This will generate an output JSON file containing aggregated tags for each of your videos.

---

### 7. Step 7: Updating the Plex Library with New Metadata

Now that we have our summarized tags stored in `metadata/`, it’s time to update our Plex Media Library with this information. We’ll do this using Plex’s API.

**Script Snippet: `update_plex.py`**

```python
import requests

def update_plex_metadata(video_id, library_section_id, plex_token, title=None, year=None, genre=None, actors=None, tags=None):
    
	"""
	Updates Plex library item with new metadata.
	
	Args:
	    video_id (str): The ID of the specific media item in Plex Library.
	    library_section_id (str): The Section ID of your Plex library where updates are to be made.
	    plex_token (str): Your unique Plex authentication token.
	    title(str): Optional updated title for media item .
	    year(int) : Optional updated release year .
	    genre(list[str]): Optional list strings representing genres applicable media item .
	    actors(list[dict]): Optional list dictionaries representing actors applicable media item .
	    tags(list[str]): Optional list strings representing descriptive/customized tagging applicable media item .

	Returns:
	   response : HTTP Response status object from requests module call made earlier towards PLEX server instance endpoint 
	   
	"""
	 
	url= f"http://localhost:32400/library/sections/{library_section_id}/all"
	headers={ "X-Plex-Token":plex_token }
	 	
	payload={}
	if title :
	     payload['title']=title 
	if year :
	     payload['year']=year 
	if genre :
	     payload['genre']=genre 
	if actors :
	     payload ['actors']=[{ "tag":actor }for actor in actors ] 
	if tags :
	     payload ['tagline']=' | '.join(tags )
	payload['type']='movie' 
	
	response=requests.put(url ,headers=headers ,json=payload )
	
	return response 

def update_all_videos_in_library(metadata_folder ,library_section_id="1", plex_token="your_plex_token_here"):

	"""
	Update every single video's corresponding metadata upon PLEX server instance based off pre-generated aggregations housed within designated directory .

	Args : 
	   meta data folder str Path pointing towards housing location encompassing pre-generated aggregation result for each corresponding analyzed movie/clip via RESNET deep convolutional network architecture previously . 

	   library section id str PLEX Library section ID wherein all updates shall commence towards respective items found therein . 

	   plex token str Authentication Token string value associated current authorized PLEX user account managing upcoming changes remotely across server instance respectively . 

	Returns : None Output console-based feedback/status report per completed operation cycle successfully updating each identified entry accordingly!

	"""

	for meta_file in sorted(os.listdir(metadata_folder)):
		
		 if meta_file.endswith(".json"):
		      with open(os.path.join(metadata_folder ,meta_file)) as jfile :
		      	meta=json.load(jfile )
		      	video name=os.path.splitext(meta ["video"])[0]
		      	file name=os.path.basename(meta_file ).replace("_meta data .json ","")
		      	print ("Updating :",video name,"....")
		      	resp=update plex meta data(video_name ,
		              	library section id ,
		              	        plex token ,
		              	        title=file name ,
		              	        genre=["Custom"],
		              	        year=2022,
		               	        	tags=list(meta["tags"].keys()))
		        if resp.status_code==200 :print ("Successfully Updated :",video name)
		        else :print ("Failed Update :",resp.text)


if __name__ == "__main__":
	update_all_videos_in_library('metadata/', "1", "your_plex_token_here")
```
 


**How It Works**:


• First ,this code extracts out relevant information gathered earlier during previous stages then subsequently maps over corresponding entries found residing inside designated METADATA subfolder finally proceeds onwards updating individual film records held within target plex database accordingly! 


Run following command :

```bash

python update _plex.py

```
 


This should now populate respective entires residing inside designated library section under current active account alongside any newly added descriptive custom labels derived directly through preceding analysis pipeline steps covered above! 


---

### Conclusion


Congratulations ! You've successfully implemented automated solution leveraging RESNET capable transforming raw pixel data extracted directly out films ultimately mapped useful descriptive labeling suitable driving intelligent search indexing features built across modern streaming platforms like PLEX MEDIA SERVER instances globally today while minimizing manual intervention overhead required previously!!!