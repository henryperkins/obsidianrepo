```python
import os
import sys
import argparse
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext

def rename_and_modify_py_to_md(directory, recursive=False, dry_run=False, log_callback=None):
    """
    Renames all .py files in the specified directory to .md and wraps their contents in Markdown code fences.

    Args:
        directory (str): The path to the directory to process.
        recursive (bool): If True, process directories recursively.
        dry_run (bool): If True, perform a dry run without renaming or modifying files.
        log_callback (callable): Function to call with log messages.
    """
    if not os.path.isdir(directory):
        msg = f"Error: The path '{directory}' is not a valid directory."
        if log_callback:
            log_callback(msg)
        else:
            print(msg)
        return

    # Choose the appropriate walker based on recursion
    if recursive:
        walker = os.walk(directory)
    else:
        try:
            files = os.listdir(directory)
        except OSError as e:
            msg = f"Error accessing directory '{directory}': {e}"
            if log_callback:
                log_callback(msg)
            else:
                print(msg)
            return
        walker = [(directory, [], files)]

    for root, dirs, files in walker:
        for filename in files:
            if filename.lower().endswith('.py'):
                old_path = os.path.join(root, filename)
                new_filename = os.path.splitext(filename)[0] + '.md'
                new_path = os.path.join(root, new_filename)

                # Check if the new file name already exists to avoid overwriting
                if os.path.exists(new_path):
                    msg = f"Skipping '{old_path}': '{new_filename}' already exists."
                    if log_callback:
                        log_callback(msg)
                    else:
                        print(msg)
                    continue

                if dry_run:
                    msg = f"[Dry Run] Would rename: '{old_path}' -> '{new_path}' and modify contents."
                    if log_callback:
                        log_callback(msg)
                    else:
                        print(msg)
                else:
                    try:
                        # Rename the file
                        os.rename(old_path, new_path)
                        msg = f"Renamed: '{old_path}' -> '{new_path}'"
                        if log_callback:
                            log_callback(msg)
                        else:
                            print(msg)

                        # Read the original content
                        with open(new_path, 'r', encoding='utf-8') as file:
                            content = file.read()

                        # Wrap the content in Markdown code fences
                        wrapped_content = f"```python\n{content}\n```"

                        # Write the modified content back to the file
                        with open(new_path, 'w', encoding='utf-8') as file:
                            file.write(wrapped_content)

                        msg = f"Modified contents of '{new_path}' to include Markdown code fences."
                        if log_callback:
                            log_callback(msg)
                        else:
                            print(msg)
                    except OSError as e:
                        msg = f"Error processing '{old_path}': {e}"
                        if log_callback:
                            log_callback(msg)
                        else:
                            print(msg)

def run_cli(args):
    """
    Executes the CLI functionality based on parsed arguments.

    Args:
        args (Namespace): Parsed command-line arguments.
    """
    rename_and_modify_py_to_md(
        directory=args.directory,
        recursive=args.recursive,
        dry_run=args.dry_run
    )

class RenameGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Rename .py to .md Files with Content Modification")
        self.create_widgets()

    def create_widgets(self):
        # Directory Selection
        dir_frame = tk.Frame(self.root)
        dir_frame.pack(pady=10, padx=10, fill='x')

        self.dir_label = tk.Label(dir_frame, text="Selected Directory:")
        self.dir_label.pack(side='left')

        self.dir_path = tk.StringVar()
        self.dir_entry = tk.Entry(dir_frame, textvariable=self.dir_path, width=50, state='readonly')
        self.dir_entry.pack(side='left', padx=5)

        self.browse_button = tk.Button(dir_frame, text="Browse", command=self.browse_directory)
        self.browse_button.pack(side='left')

        # Options
        options_frame = tk.Frame(self.root)
        options_frame.pack(pady=5, padx=10, fill='x')

        self.recursive_var = tk.BooleanVar()
        self.recursive_check = tk.Checkbutton(options_frame, text="Recursive", variable=self.recursive_var)
        self.recursive_check.pack(side='left')

        self.dry_run_var = tk.BooleanVar()
        self.dry_run_check = tk.Checkbutton(options_frame, text="Dry Run", variable=self.dry_run_var)
        self.dry_run_check.pack(side='left', padx=10)

        # Execute Button
        execute_frame = tk.Frame(self.root)
        execute_frame.pack(pady=5, padx=10, fill='x')

        self.execute_button = tk.Button(execute_frame, text="Rename and Modify Files", command=self.execute_rename)
        self.execute_button.pack()

        # Log Area
        log_frame = tk.Frame(self.root)
        log_frame.pack(pady=10, padx=10, fill='both', expand=True)

        self.log_text = scrolledtext.ScrolledText(log_frame, state='disabled', height=15)
        self.log_text.pack(fill='both', expand=True)

    def browse_directory(self):
        directory = filedialog.askdirectory()
        if directory:
            self.dir_path.set(directory)

    def execute_rename(self):
        directory = self.dir_path.get()
        if not directory:
            messagebox.showwarning("No Directory Selected", "Please select a directory to proceed.")
            return

        recursive = self.recursive_var.get()
        dry_run = self.dry_run_var.get()

        # Disable the execute button to prevent multiple clicks
        self.execute_button.config(state='disabled')
        self.log_text.config(state='normal')
        self.log_text.delete(1.0, tk.END)
        self.log_text.insert(tk.END, "Starting renaming and modification process...\n")
        self.log_text.config(state='disabled')

        # Run the renaming in a separate thread to keep the GUI responsive
        import threading
        thread = threading.Thread(target=self.run_rename, args=(directory, recursive, dry_run))
        thread.start()

    def run_rename(self, directory, recursive, dry_run):
        rename_and_modify_py_to_md(
            directory=directory,
            recursive=recursive,
            dry_run=dry_run,
            log_callback=self.update_log
        )
        self.update_log("Operation completed.")
        # Re-enable the execute button
        self.execute_button.config(state='normal')

    def update_log(self, message):
        self.log_text.config(state='normal')
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state='disabled')

def run_gui():
    root = tk.Tk()
    app = RenameGUI(root)
    root.mainloop()

def main():
    parser = argparse.ArgumentParser(description="Rename all .py files in a directory to .md and modify contents.")
    parser.add_argument("directory", nargs='?', help="Path to the target directory")
    parser.add_argument("-r", "--recursive", action="store_true",
                        help="Recursively rename .py files in subdirectories")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be renamed and modified without making any changes")
    parser.add_argument("--gui", action="store_true",
                        help="Launch the graphical user interface")

    args = parser.parse_args()

    if args.gui:
        run_gui()
    elif args.directory:
        run_cli(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
```