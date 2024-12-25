I apologize for the error in the script. It seems the `sed` command wasn't formatted correctly. Let's modify the script to fix this issue and make it more robust. Here's an updated version:​​​​​​​​​​​​​​​​

To use this updated script:

1. Save it to a file, e.g., `remove_anaconda.sh`
2. Make it executable: `chmod +x remove_anaconda.sh`
3. Run it: `./remove_anaconda.sh`

This script should work more reliably. It:

1. Checks if conda is installed before trying to deactivate environments.
2. Checks for the existence of directories before trying to remove them.
3. Provides more informative output about what it's doing.
4. Uses `sed` correctly to modify shell configuration files.

After running this script, Anaconda should be removed from your system. Remember to restart your terminal or source your shell configuration file after running the script.

If you encounter any issues or have any questions about what the script is doing, please let me know, and I'll be happy to explain further or help troubleshoot.​​​​​​​​​​​​​​​​

---
The error you're encountering indicates that GitHub is denying access to the repository because it can't authenticate your connection. This is typically due to SSH key issues. Let's troubleshoot this step by step:

1. Verify your SSH key:
   Make sure you have an SSH key generated and added to your GitHub account.

   a. Check for existing SSH keys:
      ```
      ls -al ~/.ssh
      ```
   
   b. If you don't see any files named `id_rsa.pub` or `id_ed25519.pub`, you need to generate a new SSH key:
      ```
      ssh-keygen -t ed25519 -C "your_email@example.com"
      ```

2. Add your SSH key to the ssh-agent:
   ```
   eval "$(ssh-agent -s)"
   ssh-add ~/.ssh/id_ed25519
   ```

3. Add your SSH key to your GitHub account:
   a. Copy your public key:
      ```
      cat ~/.ssh/id_ed25519.pub
      ```
   b. Go to GitHub > Settings > SSH and GPG keys > New SSH key
   c. Paste your key and save

4. Test your SSH connection:
   ```
   ssh -T git@github.com
   ```
   You should see a message like "Hi username! You've successfully authenticated..."

5. Verify your remote URL:
   ```
   git remote -v
   ```
   Ensure it starts with `git@github.com:` not `https://github.com/`

6. If the remote URL is incorrect, change it:
   ```
   git remote set-url origin git@github.com:username/repository.git
   ```

7. Try pushing again:
   ```
   git push --set-upstream origin main
   ```

If you're still having issues, there are a few more things to check:

8. Ensure you're a collaborator on the repository if it's not your own.

9. Check if you're behind a firewall that might be blocking SSH connections.

10. Verify that you're using the correct GitHub account if you have multiple.

Let me know which step you get stuck on or if you need more detailed explanation for any part of this process.​​​​​​​​​​​​​​​​
