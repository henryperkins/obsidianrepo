How to install Wordpress on RHEL 9 using nginx 1.26 and PHP-FPM and MySQL 8.4 LTS 


# **Install WordPress**

## **Introduction**

WordPress is a widely-used content management system (CMS) that powers millions of websites around the globe. This guide will walk you through the steps required to install WordPress on your server.

## **Prerequisites**

Before you begin, ensure you have the following:

- A domain name and web hosting service
- Access to your web server (via FTP or control panel)
- A text editor (like Notepad++ or Sublime Text)
- An FTP client (like FileZilla)

## **Step-by-Step Guide**

### **1. Download WordPress**
First, download the latest version of WordPress from the [official website](https://wordpress.org/download/).

### **2. Extract Files**
Once downloaded, extract the files from the zip archive to a folder on your computer.

### ⭐**3. Upload Files to Server**
Use your FTP client to upload all files from the extracted WordPress folder to your web server's root directory (usually `public_html`).

### **4. Create a Database**
Log in to your web hosting control panel and create a new MySQL database and user with full privileges.

#### Steps:
  - Go to MySQL Databases section.
  - Create a new database.
  - Create a new user and assign this user to the database with full permissions.
  - Note down the database name, username, and password for later use.

### ⭐**5. Configure wp-config.php**
Rename the `wp-config-sample.php` file in your uploaded directory to `wp-config.php`. Open it in your text editor and fill in your database details:

```php
define('DB_NAME', 'your_database_name');
define('DB_USER', 'your_database_user');
define('DB_PASSWORD', 'your_database_password');
define('DB_HOST', 'localhost'); // Most likely you won't need to change this
```

Save and close the file after editing.

### ⭐**6. Run the Installation Script**
Open your web browser and go to `http://yourdomain.com/wp-admin/install.php`. This will initiate the WordPress installation process.

#### Steps:
  - Choose your language.
  - Fill in site title, username, password, and email address.
  - Click “Install WordPress”.

### **7. Finalize Installation**
After successful installation, you'll see a success message with an option to log in. Use the credentials you provided during setup.

## **Conclusion**

Congratulations! You have successfully installed WordPress on your server. You can now log in to your admin dashboard at `http://yourdomain.com/wp-admin` and start customizing your site.

For further customization and advanced features, refer to [WordPress Codex](https://codex.wordpress.org/) or explore various plugins available within the admin dashboard.


> [!ai]+ AI
>
> Installing WordPress on RHEL 9 with Nginx 1.26, PHP-FPM, and MySQL 8.4 LTS involves several steps. Here’s a detailed guide to help you through the installation process:
> ### Step 1: Update the System
> First, ensure your system is up-to-date.
> ```bash
> sudo dnf update -y
> ```
> ### Step 2: Install Nginx
> Nginx might not be available in the default repositories, so you may need to add a repository for the latest version.
> ```bash
> sudo dnf install epel-release -y
> sudo dnf install nginx -y
> ```
> Enable and start Nginx:
> ```bash
> sudo systemctl enable nginx
> sudo systemctl start nginx
> ```
> ### Step 3: Install MySQL
> Add the MySQL repository:
> ```bash
> wget https://dev.mysql.com/get/mysql80-community-release-el9-1.noarch.rpm
> sudo rpm -ivh mysql80-community-release-el9-1.noarch.rpm
> ```
> Install MySQL server:
> ```bash
> sudo dnf install mysql-server -y
> ```
> Enable and start MySQL service:
> ```bash
> sudo systemctl enable mysqld
> sudo systemctl start mysqld
> ```
> Secure your MySQL installation (you’ll be prompted to set a root password and configure other security settings):
> ```bash
> mysql_secure_installation
> ```
> ### Step 4: Create a Database for WordPress
> Log into MySQL as root:
> ```bash
> mysql -u root -p
> ```
> Create a database and user for WordPress:
> ```sql
> CREATE DATABASE wordpress;
> CREATE USER 'wordpressuser'@'localhost' IDENTIFIED BY 'yourpassword';
> GRANT ALL PRIVILEGES ON wordpress.* TO 'wordpressuser'@'localhost';
> FLUSH PRIVILEGES;
> EXIT;
> ```
> ### Step 5: Install PHP and PHP-FPM
> Add the EPEL and REMI repositories to get the latest PHP versions.
> ```bash
> sudo dnf install epel-release -y
> sudo dnf install https://rpms.remirepo.net/enterprise/remi-release-9.rpm -y
> # Enable the REMI repository for PHP 8.x (use appropriate version)
> sudo dnf module reset php
> sudo dnf module enable php:remi-8.0 # or replace with your desired version like remi-8.1 etc.
> ```
> Install PHP, PHP-FPM, and necessary extensions:
> ```bash
> sudo dnf install php php-fpm php-mysqlnd php-curl php-json php-gd php-mbstring php-xml php-xmlrpc php-soap php-intl php-zip -y
> ```
> Enable and start PHP-FPM service:
> ``` bash sudo systemctl enable php-fpm sudo systemctl start php-fpm ```
> ### Step 6: Configure Nginx
> Edit the default server block or create a new configuration file for your WordPress site.
> For example:
> ``` bash sudo nano /etc/nginx/conf.d/wordpress.conf ```
> Add the following configuration:
> ``` nginx server { listen 80; server_name your_domain_or_IP; root /var/www/html/wordpress; index index.php index.html index.htm;
> location / {
> try_files $uri $uri/ /index.php?$args;
> }
> location ~ \.php$ {
> include snippets/fastcgi-php.conf;
> fastcgi_pass unix:/run/php-fpm/www.sock;
> }
> location ~ /\.ht {
> deny all;
> }
> }
> 
> Save and exit the file.
> Test Nginx configuration:
> ``` bash sudo nginx -t ```
> Reload Nginx to apply changes:
> ``` bash sudo systemctl reload nginx ```
> Step 7: Download WordPress Download latest WordPress package from official source.
> ``` bash cd /tmp curl -O https://wordpress.org/latest.tar.gz ``
> Extract downloaded package:
> ``` bash tar xzvf latest.tar.gz ``
> Move extracted files to web root directory:
> ``` bash sudo mv wordpress /var/www/html/
> Set proper permissions:
> ``` bash sudo chown -R nginx:nginx /var/www/html/wordpress sudo chmod -R 755 /var/www/html/wordpress ``
> Step 8: Complete Installation through Web Browser Open your web browser and navigate to http://your_domain_or_IP . Follow onscreen instructions to complete installation process. You will need database name , username , password which we created earlier .
> You should now have functioning WordPress site running on RHEL9 using NGINX ,PHPFPM & MYSQL .
