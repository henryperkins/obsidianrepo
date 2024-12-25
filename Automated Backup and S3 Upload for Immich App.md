```python
import os
import time
import shutil
import subprocess
import threading
import progressbar
from concurrent.futures import ThreadPoolExecutor, as_completed
import boto3
from boto3.s3.transfer import TransferConfig
from botocore.exceptions import NoCredentialsError, PartialCredentialsError, ClientError

# Environment Variables
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

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".flv", ".wmv", ".webm",
                    ".ogg", ".ogv", ".m4v", ".3gp", ".mxf", ".rm", ".swf",
                    ".f4v", ".ts", ".m2ts", ".mts", ".dv", ".vp9", ".mpeg", ".mpg"}

# Log Message Function
def log_message(message):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}")

# Requirement Check Function
def check_requirements():
    missing_tools = []
    for tool in ["aws", "docker"]:
        if not shutil.which(tool):
            missing_tools.append(tool)
    if missing_tools:
        log_message(f"Error: Missing tools: {', '.join(missing_tools)}. Please install them and try again.")
        exit(1)

# Directory Size Calculation
def calculate_directory_size(directory):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        for f in filenames:
            if os.path.splitext(f)[1].lower() in VIDEO_EXTENSIONS:
                fp = os.path.join(dirpath, f)
                total_size += os.path.getsize(fp)
    return total_size

# File Copy Function with Progress Bar
def copy_file(source, destination, bar, lock):
    shutil.copy2(source, destination)
    with lock:
        bar.update(bar.value + os.path.getsize(source))

# Backup Directory Function
def backup_video_files(source_dir, dest_dir, dir_name, max_workers=4):
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
                if os.path.splitext(filename)[1].lower() in VIDEO_EXTENSIONS:
                    source_path = os.path.join(dirpath, filename)
                    dest_path = os.path.join(dest_dir, os.path.relpath(source_path, source_dir))
                    futures.append(executor.submit(copy_file, source_path, dest_path, bar, lock))
        
        for future in as_completed(futures):
            future.result()

    bar.finish()
    log_message(f"Finished backup of {dir_name}")

# PostgreSQL Database Backup Function
def backup_postgresql_database():
    log_message("Backing up database...")
    dump_command = f"docker compose exec -T {DB_CONTAINER} pg_dump -U {DB_USER} {DB_NAME}"
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

# S3 Bucket Check/Create Function
def ensure_s3_bucket_exists(s3_client, bucket_name):
    try:
        s3_client.head_bucket(Bucket=bucket_name)
        log_message(f"S3 bucket {bucket_name} already exists.")
    except ClientError:
        log_message(f"S3 bucket {bucket_name} does not exist. Creating new bucket...")
        try:
            s3_client.create_bucket(Bucket=bucket_name, CreateBucketConfiguration={
                'LocationConstraint': s3_client.meta.region_name})
            s3_client.put_bucket_policy(
                Bucket=bucket_name,
                Policy='''{
                    "Version": "2012-10-17",
                    "Statement": [
                        {
                            "Effect": "Allow",
                            "Principal": "*",
                            "Action": "s3:GetObject",
                            "Resource": "arn:aws:s3:::%s/*"
                        }
                    ]
                }''' % bucket_name
            )
            s3_client.put_bucket_encryption(
                Bucket=bucket_name,
                ServerSideEncryptionConfiguration={
                    'Rules': [
                        {
                            'ApplyServerSideEncryptionByDefault': {
                                'SSEAlgorithm': 'AES256'
                            }
                        },
                    ]
                }
            )
            log_message(f"S3 bucket {bucket_name} created with proper permissions and configuration.")
        except ClientError as e:
            log_message(f"Error creating S3 bucket: {e}")
            exit(1)

# S3 Upload Function with Progress Bar
def upload_file_to_s3(local_path, s3_bucket, s3_prefix, lock, bar):
    try:
        s3 = boto3.client('s3')
        relative_path = os.path.relpath(local_path, LOCAL_BACKUP_DIR)
        s3_path = os.path.join(s3_prefix, relative_path).replace("\\", "/")
        config = TransferConfig(multipart_threshold=1024 * 25, max_concurrency=10, multipart_chunksize=1024 * 25, use_threads=True)

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

# Main Script Execution
if __name__ == "__main__":
    check_requirements()

    if not os.path.exists(LOCAL_BACKUP_DIR):
        os.makedirs(LOCAL_BACKUP_DIR)

    # Ensure the S3 bucket exists or create it
    s3 = boto3.client('s3')
    ensure_s3_bucket_exists(s3, S3_BUCKET)

    log_message("Stopping Immich services...")
    subprocess.run(["docker", "compose", "down"], cwd=IMMICH_DIR)

    backup_video_files(LIBRARY_DIR, os.path.join(LOCAL_BACKUP_DIR, "library"), "Video Files")

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
    