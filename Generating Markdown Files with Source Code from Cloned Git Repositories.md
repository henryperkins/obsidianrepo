Certainly! To format the source code into code blocks in the Markdown file, you can modify the `generate_markdown_file` function to include code blocks for each file. Here's an updated version of the script that does this:

```python
from git import Repo
import os

def clone_repository(repo_url, target_dir):
    """
    Clones a Git repository to a specified directory.

    :param repo_url: URL of the Git repository
    :param target_dir: Directory where the repository will be cloned
    """
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    try:
        Repo.clone_from(repo_url, target_dir)
        print(f"Repository cloned successfully to {target_dir}")
        return True
    except Exception as e:
        print(f"Error cloning repository: {e}")
        return False

def generate_markdown_file(target_dir, markdown_file):
    """
    Generates a Markdown file listing the files and directories in the target directory,
    and includes the content of each file in a code block.

    :param target_dir: Directory to list
    :param markdown_file: Path to the Markdown file to be generated
    """
    with open(markdown_file, 'w') as f:
        f.write(f"# Repository Contents\n\n")
        f.write(f"## Directory: {target_dir}\n\n")
        
        for root, dirs, files in os.walk(target_dir):
            level = root.replace(target_dir, '').count(os.sep)
            indent = ' ' * 4 * (level)
            f.write(f"{indent}**{os.path.basename(root)}/**\n")
            
            sub_indent = ' ' * 4 * (level + 1)
            for file in files:
                file_path = os.path.join(root, file)
                f.write(f"{sub_indent}- {file}\n")
                f.write(f"{sub_indent}  ```\n")
                
                with open(file_path, 'r', encoding='utf-8') as file_content:
                    for line in file_content:
                        f.write(f"{sub_indent}  {line}")
                
                f.write(f"{sub_indent}  ```\n")
                f.write("\n")

if __name__ == "__main__":
    # URL of the Git repository
    repo_url = "https://github.com/username/repository.git"
    
    # Directory where the repository will be cloned
    target_dir = "./cloned_repo"
    
    # Path to the Markdown file to be generated
    markdown_file = "./repository_contents.md"
    
    if clone_repository(repo_url, target_dir):
        generate_markdown_file(target_dir, markdown_file)
        print(f"Markdown file generated at {markdown_file}")
```

### Explanation

1. **Import the necessary modules**:
   - `Repo` from `git` is used to interact with Git repositories.
   - `os` is used to handle directory and file operations.

2. **Define the `clone_repository` function**:
   - This function takes two parameters: `repo_url` (the URL of the Git repository) and `target_dir` (the directory where the repository will be cloned).
   - It checks if the target directory exists and creates it if it doesn't.
   - It attempts to clone the repository using `Repo.clone_from` and prints a success message if the cloning is successful.
   - If an error occurs during cloning, it catches the exception and prints an error message.
   - It returns `True` if the cloning is successful, otherwise `False`.

3. **Define the `generate_markdown_file` function**:
   - This function takes two parameters: `target_dir` (the directory to list) and `markdown_file` (the path to the Markdown file to be generated).
   - It opens the Markdown file for writing and writes a header and a list of files and directories in the target directory.
   - It uses `os.walk` to traverse the directory tree and writes the directory and file names to the Markdown file.
   - For each file, it reads the content and writes it to the Markdown file within a code block.

4. **Main block**:
   - Set the `repo_url` to the URL of the Git repository you want to clone.
   - Set the `target_dir` to the directory where you want to clone the repository.
   - Set the `markdown_file` to the path where you want to generate the Markdown file.
   - Call the `clone_repository` function with the specified `repo_url` and `target_dir`.
   - If the cloning is successful, call the `generate_markdown_file` function to generate the Markdown file.

### Example Usage

If you run the script with the following values:

```python
repo_url = "https://github.com/username/repository.git"
target_dir = "./cloned_repo"
markdown_file = "./repository_contents.md"
```

The source code from the repository at `https://github.com/username/repository.git` will be cloned into the `./cloned_repo` directory, and a Markdown file named `repository_contents.md` will be generated in the current directory. The Markdown file will list the contents of the cloned repository and include the content of each file in a code block.