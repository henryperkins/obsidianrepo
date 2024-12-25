### White Paper / Spec Sheet for Automated Backup Script

---

#### Title: Automated Backup and Upload Script for Immich Application

##### Author: [Your Name]

---

### Overview

This document outlines the features, functions, and purpose of the automated backup and upload script designed for the Immich application. The script is crafted to automate the backup of photo libraries and PostgreSQL databases, ensuring data integrity and availability by uploading the backups to an Amazon S3 bucket.

---

### Purpose

The primary purpose of this script is to provide a reliable and automated solution for backing up the Immich application's photo library and PostgreSQL database. This ensures that valuable data is preserved and can be restored in case of system failure, data corruption, or other unforeseen events.

---

### Features and Functions

#### 1. Environment Configuration
- **Environment Variables:**
  - `IMMICH_DIR`: Base directory for the Immich application.
  - `LIBRARY_DIR`: Directory path for the photo library.
  - `POSTGRES_DIR`: Directory path for PostgreSQL data.
  - `DB_CONTAINER`: Name of the PostgreSQL Docker container.
  - `DB_USER`: PostgreSQL username.
  - `DB_NAME`: PostgreSQL database name.
  - `BACKUP_DATE`: Timestamp for the current backup.
  - `S3_BUCKET`: Name of the Amazon S3 bucket for storing backups.
  - `S3_PREFIX`: S3 path prefix for the backup files.
  - `LOCAL_BACKUP_DIR`: Local directory path for storing temporary backups.

#### 2. Logging
- **Log Messages:**
  - Function to log messages with timestamps to track script progress and status.

#### 3. Requirement Checks
- **Dependency Verification:**
  - Checks for the presence of essential tools (`aws` and `docker`) and exits with an error message if any tool is missing.

#### 4. Backup Process
- **Library and PostgreSQL Data Backup:**
  - Recursively copies files from the source directories to a local backup directory.
  - Uses multi-threading to optimize the copy process, improving speed and efficiency.
  - Displays a progress bar to monitor the backup status in real-time.

- **PostgreSQL Database Backup:**
  - Dumps the PostgreSQL database using `pg_dump`.
  - Compresses the dump file using `gzip`.
  - Displays a progress bar to monitor the database backup status.

#### 5. Upload to Amazon S3
- **Optimized S3 Upload:**
  - Configures the S3 transfer using `boto3` with an optimized `TransferConfig` for multipart uploads and high concurrency.
  - Recursively uploads files from the local backup directory to the specified S3 bucket.
  - Utilizes multi-threading to upload multiple files simultaneously.
  - Displays a progress bar to monitor the upload status in real-time.

#### 6. Docker Management
- **Docker Container Management:**
  - Stops Immich services by bringing down the Docker Compose stack.
  - Starts the PostgreSQL container to perform the database dump.
  - Restarts the entire Immich services stack after the backup process is completed.

#### 7. Cleanup
- **Local Backup Cleanup:**
  - Removes the local backup directory after the backup and upload process is completed to free up disk space.

---

### Detailed Script Breakdown

#### 1. Environment Variables Setup
```python
IMMICH_DIR = "/home/azureuser/immich-app"
LIBRARY_DIR = f"{IMMICH_DIR}/library"
POSTGRES_DIR = f"{IMMICH_DIR}/postgres"
DB_CONTAINER = "immich_postgres"
DB_USER = "postgres"
DB_NAME = "immich"
BACKUP_DATE = time.strftime("%Y%m%d_%H%M%S")
S3_BUCKET = "immich-backup-bucket"
S3_PREFIX = f"immich-backup/Immich_Backup_{BACKUP_DATE}"
LOCAL_BACKUP_DIR = f"/tmp/immich_backup_{BACKUP_DATE}"
```

#### 2. Log Message Function
```python
def log_message(message):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}")
```

#### 3. Requirement Check Function
```python
def check_requirements():
    missing_tools = []
    for tool in ["aws", "docker"]:
        if not shutil.which(tool):
            missing_tools.append(tool)
    if missing_tools:
        log_message(f"Error: Missing tools: {', '.join(missing_tools)}. Please install them and try again.")
        exit(1)
```

#### 4. Directory Size Calculation
```python
def calculate_directory_size(directory):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size
```

#### 5. File Copy Function with Progress Bar
```python
def copy_file(source, destination, bar, lock):
    shutil.copy2(source, destination)
    with lock:
        bar.update(bar.value + os.path.getsize(source))
```

#### 6. Backup Directory Function
```python
def backup_directory(source_dir, dest_dir, dir_name, max_workers=4):
    if not os.path.isdir(source_dir):
        log_message(f"Error: Source directory {source_dir} does not exist")
        return 1

    log_message(f"Starting backup of {dir_name}...")
    total_size = calculate_directory_size(source_dir)
    widgets = [
        f"{dir_name}: ", progressbar.Percentage(), ' ',
        progressbar.Bar(), ' ',
        progressbar.ETA(), ' ',
        progressbar.FileTransferSpeed(), ' ',
        progressbar.DataSize(), '/', progressbar.DataSize(variable='max_value')
    ]
    bar = progressbar.ProgressBar(maxval=total_size, widgets=widgets)
    bar.start()

    futures = []
    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for dirpath, dirnames, filenames in os.walk(source_dir):
            for dirname in dirnames:
                dest_path = os.path.join(dest_dir, os.path.relpath(os.path.join(dirpath, dirname), source_dir))
                os.makedirs(dest_path, exist_ok=True)
            for filename in filenames:
                source_path = os.path.join(dirpath, filename)
                dest_path = os.path.join(dest_dir, os.path.relpath(source_path, source_dir))
                futures.append(executor.submit(copy_file, source_path, dest_path, bar, lock))
        
        for future in as_completed(futures):
            future.result()

    bar.finish()
    log_message(f"Finished backup of {dir_name}")
```

#### 7. PostgreSQL Database Backup Function
```python
def backup_postgresql_database():
    log_message("Backing up database...")
    dump_command = f"docker compose exec -T database pg_dump -U {DB_USER} {DB_NAME}"
    gzip_command = f"gzip > {LOCAL_BACKUP_DIR}/immich_db_backup.sql.gz"
    total_size_estimate = 1024 * 1024 * 100  # 100 MB estimate for progress bar (adjust as needed)

    widgets = [
        "Database: ", progressbar.Percentage(), ' ',
        progressbar.Bar(), ' ',
        progressbar.ETA(), ' ',
        progressbar.FileTransferSpeed(), ' ',
        progressbar.DataSize(), '/', progressbar.DataSize(variable='max_value')
    ]
    bar = progressbar.ProgressBar(maxval=total_size_estimate, widgets=widgets)
    bar.start()

    with subprocess.Popen(dump_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as proc_dump, \
         subprocess.Popen(gzip_command, shell=True, stdin=proc_dump.stdout, stdout=subprocess.PIPE) as proc_gzip:
        while proc_gzip.poll() is None:
            time.sleep(0.1)
            bar.update(min(bar.value + 1024 * 100, total_size_estimate))  # Increment by 100 KB

    bar.finish()
```

#### 8. S3 Upload Function with Progress Bar
```python
def upload_file_to_s3(local_path, s3_bucket, s3_prefix, lock, bar):
    try:
        s3 = boto3.client('s3')
        relative_path = os.path.relpath(local_path, LOCAL_BACKUP_DIR)
        s3_path = os.path.join(s3_prefix, relative_path).replace("\\", "/")
        s3.upload_file(local_path, s3_bucket, s3_path, Config=config)
        with lock:
            bar.update(bar.value + os.path.getsize(local_path))
    except Exception as e:
        log_message(f"Error uploading {local_path} to S3: {e}")

def upload_directory_to_s3(local_dir, s3_bucket, s3_prefix, max_workers=16):
    total_size = calculate_directory_size(local_dir)
    widgets = [
        "Uploading to S3: ", progressbar.Percentage(), ' ',
        progressbar.Bar(), ' ',
        progressbar.ETA(), ' ',
        progressbar.FileTransferSpeed(), ' ',
        progressbar.DataSize(), '/', progressbar.DataSize(variable='max_value')
    ]
    bar = progressbar.ProgressBar(maxval=total_size, widgets=widgets)
    bar.start()

    futures = []
    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for root, _, files in os.walk(local_dir):
            for file in files:
                local_path = os.path.join(root, file)
                futures.append(executor.submit(upload_file_to_s3, local_path, s3_bucket, s3_prefix, lock, bar))
        
        for future in as_completed(futures):
            future.result()

    bar.finish()
```
9. **Main Script Execution**
```python
if __name__ == "__main__":
    check_requirements()

    if not os.path.exists(LOCAL_BACKUP_DIR):
        os.makedirs(LOCAL_BACKUP_DIR)

    log_message("Stopping Immich services...")
    subprocess.run(["docker", "compose", "down"], cwd=IMMICH_DIR)

    backup_directory(LIBRARY_DIR, os.path.join(LOCAL_BACKUP_DIR, "library"), "Library")
    backup_directory(POSTGRES_DIR, os.path.join(LOCAL_BACKUP_DIR, "postgres"), "Postgres")

    log_message("Starting postgres container...")
    subprocess.run(["docker", "compose", "up", "-d", "database"], cwd=IMMICH_DIR)
    time.sleep(10)

    backup_postgresql_database()

    log_message("Starting S3 upload...")
    upload_directory_to_s3(LOCAL_BACKUP_DIR, S3_BUCKET, S3_PREFIX)

    log_message("S3 upload completed")
    log_message("S3 bucket contents:")
    subprocess.run(["aws", "s3", "ls", f"s3://{S3_BUCKET}/{S3_PREFIX}", "--recursive", "--human-readable", "--summarize"])

    log_message("Removing local backup...")
    shutil.rmtree(LOCAL_BACKUP_DIR)

    log_message("Restarting Immich services...")
    subprocess.run(["docker", "compose", "up", "-d"], cwd=IMMICH_DIR)

    log_message("Backup process completed")
```

---

### Conclusion

This script provides an automated, efficient, and optimized solution for backing up the Immich application's photo library and PostgreSQL database. With features like multi-threading for parallel file copying, progress bars for real-time monitoring, and optimized S3 uploads, this script ensures data integrity and availability while minimizing downtime and maximizing performance.