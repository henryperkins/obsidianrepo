How to configure my two EC2 instances to first catch traffic, terminate SSL/TLS (provided by certbot and OpenSSL), and then serve content from Plex Media Server. Do not produce instructions on how to install NGINX Plus or Plex. Provide detailed, accurate instruction of each file edit, without glossing over small configuration processes. Include full and complete conf files for nginx as well as steps to ensure files and directories are properly created / defined and permissible. Save your tokens for producing the step-by-step tutorial, and no intro repeating this prompt. The domain is umademe.ink, the public IP of the reverse proxy is 3.137.38.193 and the ip of the instance with plex is 3.15.87.194 

Certainly! Below are the detailed instructions and configuration files to set up an NGINX Plus reverse proxy with SSL/TLS termination and caching for a Plex Media Server on Ubuntu. These instructions will assume that you already have NGINX Plus and Plex installed on their respective EC2 instances.

### Step-by-Step Tutorial:

#### 1. Install Certbot for SSL/TLS:

First, install Certbot on your reverse proxy server (IP: 3.137.38.193).

```sh
sudo apt update
sudo apt install certbot
```

#### 2. Obtain SSL Certificates:

Use Certbot to obtain a certificate for your domain (`umademe.ink`).

```sh
sudo certbot certonly --standalone -d umademe.ink
```

This will place your certificates in `/etc/letsencrypt/live/umademe.ink`.

#### 3. Configure NGINX Plus:

Edit the NGINX configuration file to set up the reverse proxy with SSL termination and caching.

Create the directory structure for the cache:

```sh
sudo mkdir -p /var/cache/nginx/cache_zone
sudo chown -R www-data:www-data /var/cache/nginx/cache_zone
```

Edit or create the main configuration file `/etc/nginx/nginx.conf`:

```nginx
user www-data;
worker_processes auto;
pid /run/nginx.pid;

events {
    worker_connections 1024;
}

http {
    include       /etc/nginx/mime.types;
    default_type  application/octet-stream;

    log_format main '$remote_addr - $remote_user [$time_local] "$request" '
                      '$status $body_bytes_sent "$http_referer" '
                      '"$http_user_agent" "$http_x_forwarded_for"';

    access_log /var/log/nginx/access.log main;

    sendfile        on;
    keepalive_timeout 65;

    # Define a cache zone for static content
    proxy_cache_path /var/cache/nginx/cache_zone levels=1:2 keys_zone=my_cache:10m max_size=1g inactive=60m use_temp_path=off;

    include /etc/nginx/conf.d/*.conf;
}
```

Create or edit the reverse proxy configuration file `/etc/nginx/conf.d/reverse-proxy.conf`:

```nginx
server {
    listen 80;
    server_name umademe.ink;

    location /.well-known/acme-challenge/ {
        root /var/www/certbot;
    }

    location / {
        return 301 https://$host$request_uri;
    }
}

server {
    listen 443 ssl http2;
    server_name umademe.ink;

    ssl_certificate /etc/letsencrypt/live/umademe.ink/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/umademe.ink/privkey.pem;

    ssl_protocols TLSv1 TLSv1.1 TLSv1.2 TLSv1.3;
    
	ssl_ciphers "ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384";
	ssl_prefer_server_ciphers on; 
	ssl_session_cache shared:SSL:10m; 

	client_max_body_size 100M; 

	add_header Strict-Transport-Security "max-age=63072000; includeSubDomains; preload";

	location ~* \.(jpg|jpeg|png|gif|ico|css|js)$ {
	    expires max; 
	    add_header Cache-Control "public, no-transform"; 
	    access_log off; 
	}

	location / {
	    proxy_pass http://3.15.87.194:32400; # Replace with Plex Media Server's internal IP and port.
	    proxy_set_header Host $host;
	    proxy_set_header X-Real-IP $remote_addr;
	    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
	    proxy_set_header X-Forwarded-Proto $scheme;

	    # Enable caching of responses from Plex Media Server.
	    proxy_cache my_cache;
            proxy_cache_valid any 30m; # Adjust cache duration as needed.
            add_header X-Proxy-Cache $upstream_cache_status;

	    # Buffer settings to handle large files efficiently.
	    client_max_body_size 0;
            client_body_buffer_size 128k;
            proxy_buffering on;
            proxy_buffers 32 4k;
            proxy_busy_buffers_size 64k;
            proxy_temp_file_write_size 64k;

	}
}
```

#### Final Steps:

1. **Test Configuration**:
   
   Test the configuration to ensure there are no syntax errors.

   ```sh
   sudo nginx -t
   ```

2. **Restart NGINX**:
   
   Restart NGINX to apply changes.

   ```sh
   sudo systemctl restart nginx
   ```

3. **Set Up Renewals**:
   
   Ensure your SSL certificates renew automatically by adding a cron job.

   ```sh
   echo "0 */12 * * * root certbot renew --quiet && systemctl reload nginx" | sudo tee -a /etc/crontab > /dev/null
   ```

### File Permissions and Ownership

Ensure that all necessary directories have appropriate permissions and ownerships:

```sh
# Change ownership of the webroot directory used by Certbot if required.
sudo mkdir -p /var/www/certbot 
sudo chown www-data:www-data -R /var/www/certbot

# Permissions for Let's Encrypt directories (usually handled by certbot).
sudo chmod -R go-rwx /etc/letsencrypt/{live,archive}
```
This concludes setting up an NGINX Plus reverse-proxy with SSL/TLS termination and caching for Plex Media Server traffic routed through umademe.ink domain from one EC2 instance to another!