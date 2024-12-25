`sudo mkdir -p /etc/ssl/nginx`

`sudo apt-get update && sudo`

`apt-get install apt-transport-https lsb-release ca-certificates wget gnupg2`

`sudo wget https://cs.nginx.com/static/keys/nginx_signing.key && sudo apt-key add nginx_signing.key`

`sudo wget https://cs.nginx.com/static/keys/app-protect-security-updates.key && sudo apt-key add app-protect-security-updates.key`

`printf "deb https://pkgs.nginx.com/plus/ubuntu lsb_release -cs nginx-plus\n" | sudo tee /etc/apt/sources.list.d/nginx-plus.list`

`printf "deb https://pkgs.nginx.com/app-protect/ubuntu lsb_release -cs nginx-plus\n" | sudo tee /etc/apt/sources.list.d/nginx-app-protect.list`


`printf "deb https://pkgs.nginx.com/app-protect-security-updates/ubuntu lsb_release -cs nginx-plus\n" | sudo tee /etc/apt/sources.list.d/app-protect-security-updates.list`

`sudo wget -P /etc/apt/apt.conf.d https://cs.nginx.com/static/files/90pkgs-nginx`

`sudo apt-get update`
`sudo apt-get install app-protect`

`sudo mkdir -p /etc/docker/certs.d/private-registry.nginx.com`
`sudo cp /etc/ssl/nginx/nginx-repo.crt /etc/docker/certs.d/private-registry.nginx.com/client.cert`
`sudo cp /etc/ssl/nginx/nginx-repo.key /etc/docker/certs.d/private-registry.nginx.com/client.key`