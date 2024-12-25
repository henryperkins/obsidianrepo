## Dropbox & OneDrive Linux Install
### Dropbox & OneDrive Linux Install
#### **OneDrive Installation on Ubuntu Server**

1) **Install Dependencies:**

   ```bash
   sudo apt update
   sudo apt install build-essential libcurl4-openssl-dev libsqlite3-dev pkg-config git
   ```

2) **Install the D Compiler:**

   ```bash
   sudo apt install ldc
   ```

3) **Clone the OneDrive Repository:**

   ```bash
   git clone https://github.com/abraunegg/onedrive.git
   cd onedrive
   ```

4) **Generate the Makefile and Build the Client:**

   ```bash
   ./configure
   make
   sudo make install
   ```

5) **Authenticate with OneDrive:**

   - Run the `onedrive` command to start the authentication process:

	 ```bash
     onedrive
     ```

   - Copy the URL provided in the terminal, paste it into a browser, and log in with your Microsoft account.
   - After granting access, you may be redirected to a URL that starts with `https://login.microsoftonline.com/common/oauth2/nativeclient?code=...`. Copy this URL.
   - Back in the terminal, use the `--auth-response` option with the copied URL:

	 ```bash
     onedrive --auth-response "https://login.microsoftonline.com/common/oauth2/nativeclient?code=..."
     ```

6) **Synchronize Files:**

   ```bash
   onedrive --synchronize
   ```

7) **Continuous Monitoring (Optional):**

   If you want continuous synchronization, use:

   ```bash
   onedrive --monitor
   ```

---

#### **Dropbox Installation on Ubuntu Server**

1) **Install Dropbox:**

   ```bash
   sudo apt update
   sudo apt install python3-gpg dropbox
   ```

2) **Download and Install the Dropbox Daemon:**

   ```bash
   cd ~ && wget -O - "https://www.dropbox.com/download?plat=lnx.x86_64" | tar xzf -
   ```

3) **Run Dropbox for Initial Setup:**

   ```bash
   ~/.dropbox-dist/dropboxd
   ```

   - This command will generate a URL. Open the URL in a browser and log in to link your Dropbox account to your server.

4) **Install the Dropbox CLI (Optional):**

   ```bash
   sudo apt install python3-pip
   pip install dropbox-cli
   ```

5) **Configure Dropbox to Sync to a Specific Directory:**

   By default, Dropbox syncs to `~/Dropbox`. If you want to change the sync directory to something like `/mnt/media/`:

   ```bash
   dropbox start -i
   ```

   After starting Dropbox, stop it and move the Dropbox directory:

   ```bash
   dropbox stop
   mv ~/Dropbox /mnt/media/Dropbox
   ln -s /mnt/media/Dropbox ~/Dropbox
   dropbox start
   ```

This process should cover both the OneDrive and Dropbox setups on your Ubuntu server. Let me know if you need further assistance!