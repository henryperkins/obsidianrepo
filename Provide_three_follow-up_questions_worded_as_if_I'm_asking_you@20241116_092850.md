---
epoch: 1731770930454
modelKey: gemini-exp-1114|google
tags:
  - copilot-conversation
---

# Provide_three_follow-up_questions_worded_as_if_I'm_asking_you@20241116_092850

**user**: Provide three follow-up questions worded as if I'm asking you based on {activeNote}?  
[Timestamp: 2024/11/16 09:28:50]

**ai**: ```python  
import hashlib  
import os  
import time  
from collections import defaultdict  
from concurrent.futures import ProcessPoolExecutor  
from pathlib import Path

import questionary  
from tqdm import tqdm

def calculate_hash(file_path, hash_algorithm='sha256'):  
    """  
    Calculates the hash of a file using the specified algorithm.

    Args:
        file_path (str): The path to the file.
        hash_algorithm (str): The hash algorithm to use (md5, sha1, sha256).

    Returns:
        str: The hexadecimal digest of the file's hash.
    """
    try:
        hasher = getattr(hashlib, hash_algorithm)()
        with open(file_path, 'rb') as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception as e:
        print(f"Error calculating hash for {file_path}: {e}")
        return None

def get_video_metadata(file_path):  
    """  
    Retrieves basic metadata of a video file (size, extension).

    Args:
        file_path (str): The path to the file.

    Returns:
        dict: A dictionary containing metadata (size, extension).
    """
    try:
        stat_info = os.stat(file_path)
        return {
            'size': stat_info.st_size,
            'extension': Path(file_path).suffix.lower()
        }
    except Exception as e:
        print(f"Error getting metadata for {file_path}: {e}")
        return None

def find_duplicate_videos(directory, extensions=('.mp4', '.avi', '.mov', '.mkv'), hash_algorithm='sha256'):  
    """  
    Finds duplicate video files in a directory based on hash and metadata.

    Args:
        directory (str): The directory to search for duplicates.
        extensions (tuple): A tuple of allowed file extensions.
        hash_algorithm (str): The hash algorithm to use.

    Returns:
        dict: A dictionary where keys are hashes and values are lists of duplicate file paths.
    """
    video_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(extensions):
                video_files.append(os.path.join(root, file))

    file_hashes = defaultdict(list)
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(calculate_hash, file, hash_algorithm): file for file in video_files}
        for future in tqdm(futures, desc="Hashing files"):
            file_path = futures[future]
            file_hash = future.result()
            if file_hash:
                file_hashes[file_hash].append(file_path)

    duplicate_files = {hash_val: files for hash_val, files in file_hashes.items() if len(files) > 1}
    
    verified_duplicates = defaultdict(list)
    for hash_val, files in duplicate_files.items():
        metadata = [get_video_metadata(file) for file in files]
        for i in range(len(files)):
            is_duplicate = False
            for j in range(i + 1, len(files)):
                if metadata[i] == metadata[j]:
                    is_duplicate = True
                    break
            if is_duplicate:
                verified_duplicates[hash_val].append(files[i])

    return {hash_val: files for hash_val, files in verified_duplicates.items() if len(files) > 1}

def handle_duplicates(duplicates, action='list'):  
    """  
    Handles the identified duplicate files based on the specified action.

    Args:
        duplicates (dict): A dictionary of duplicate files.
        action (str): The action to take (list, move, delete, dry-run).
    """
    if not duplicates:
        print("No duplicate files found.")
        return

    if action == 'list':
        print("Duplicate files found:")
        for hash_val, files in duplicates.items():
            print(f"Hash: {hash_val}")
            for file in files:
                print(f"  - {file}")
    elif action == 'move':
        target_dir = input("Enter the target directory to move duplicates: ")
        os.makedirs(target_dir, exist_ok=True)
        for hash_val, files in duplicates.items():
            for file in files[1:]:  # Keep the first file, move the rest
                os.rename(file, os.path.join(target_dir, os.path.basename(file)))
                print(f"Moved: {file} to {target_dir}")
    elif action == 'delete':
        for hash_val, files in duplicates.items():
            for file in files[1:]:  # Keep the first file, delete the rest
                os.remove(file)
                print(f"Deleted: {file}")
    elif action == 'dry-run':
        print("Dry run - Duplicates that would be processed:")
        for hash_val, files in duplicates.items():
            print(f"Hash: {hash_val}")
            for file in files[1:]:
                print(f"  - {file} (would be processed)")

def main():  
    """  
    Main function to run the duplicate video finder with interactive CLI.  
    """  
    directory = questionary.path("Enter the directory to scan for duplicate videos:").ask()  
    if not directory:  
        print("No directory provided.")  
        return

    extensions = questionary.text("Enter file extensions to consider (comma-separated, e.g., .mp4,.avi):").ask()
    extensions = tuple(ext.strip() for ext in extensions.split(',')) if extensions else ('.mp4', '.avi', '.mov', '.mkv')

    hash_algorithm = questionary.select("Select the hash algorithm:", choices=['md5', 'sha1', 'sha256']).ask()

    action = questionary.select("Select action to perform:", choices=['list', 'move', 'delete', 'dry-run']).ask()

    duplicates = find_duplicate_videos(directory, extensions, hash_algorithm)
    handle_duplicates(duplicates, action)

if **name** == "**main**":  
    main()

```

## Follow-up Questions:

1. **"Can you modify the script to also compare video duration and resolution using a library like `ffprobe` to further ensure the duplicates are truly identical?"**
2. **"How can I integrate logging into this script to keep a record of the actions taken and any errors encountered?"**
3. **"Is it possible to add a feature to generate a CSV report of the duplicate files found, including their paths, sizes, and hashes?"**

[Timestamp: 2024/11/16 09:29:40]
