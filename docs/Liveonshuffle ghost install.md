

It looks like you might be trying to either document a process or seek assistance with installing something called "Liveonshuffle ghost." However, the information provided is quite minimal, and it’s unclear what specifically you need help with.

To provide you with the best possible assistance, could you please clarify or provide more details on the following:

1. **What is "Liveonshuffle"?** 
   - Is it a software application, a plugin, or something else?

2. **What do you mean by "ghost install"?**
   - Are you referring to installing Ghost (the blogging platform)?
   - Or is there another context in which you're using the term "ghost"?

3. **What are "Starred Blocks"?**
   - Are these specific configurations or settings?
   
4. **Any specific steps you're stuck on?**

If you're looking to install Ghost (the blogging platform), here are some basic steps to get started:

### Installing Ghost

1. **Install Node.js:**
   Make sure Node.js is installed on your system.

   ```bash
   curl -sL https://deb.nodesource.com/setup_14.x | sudo -E bash -
   sudo apt-get install -y nodejs
   ```

2. **Install Ghost-CLI:**
   The Ghost-CLI tool helps manage your Ghost install.

   ```bash
   sudo npm install ghost-cli@latest -g
   ```

3. **Create a directory for your blog and navigate into it:**

   ```bash
   mkdir my-ghost-blog
   cd my-ghost-blog
   ```

4. **Install Ghost:**

    ```bash
    ghost install
    ```

5. Follow the prompts which will guide you through the setup process including configuring MySQL and setting up your domain.

### Additional Information Needed

If this is not what you were looking for, please provide additional information so I can better assist you:

- Any error messages you're encountering.
- Specific configurations or environment details.
  
Feel free to expand on any points that would help clarify your request!```bash
ghost install --db mysql --port 2368 \
--dbhost 127.0.0.1 \
--dbuser $DATABASE_USER_NAME \
--dbpass $DATABASE_USER_PASSWORD \
--dbname $DATABASE_NAME \
--process local \
--no-setup-linux-user \
--no-setup-ssl \
--no-setup-nginx \
--dir /home/$SITE_USER/htdocs/$DOMAIN/
```