# Deployment Guide: Project Alpha (Rocky Linux 9)

This guide details how to deploy Project Alpha to a production Linux server (VPS/VM) using Docker Compose on **Rocky Linux 9**.

## 1. Prerequisites
- **Server**: A Linux VPS running Rocky Linux 9 (or similar RHEL-based distro).
  - Minimum specs: 2 vCPU, 4GB RAM (due to ML models).
- **Domain (Optional)**: If you plan to expose the application via HTTP.
- **SSH Key**: For secure server access.

## 2. Server Provisioning & Security
Before deploying the application, secure the server.

### 2.1 Login and Update
```bash
ssh root@your_server_ip
dnf upgrade -y
```

### 2.2 Create a Non-Root User
Do not run the application as root. We will create a user named `deploy` and add them to the `wheel` group for sudo access.
```bash
adduser deploy
passwd deploy  # Set a password for the deploy user
usermod -aG wheel deploy
su - deploy
```

### 2.3 Configure Firewall (firewalld)
Rocky Linux uses `firewalld` by default. Allow only SSH (22).
```bash
sudo systemctl start firewalld
sudo systemctl enable firewalld
sudo firewall-cmd --permanent --add-service=ssh
# If you plan to add a web interface later, uncomment the following:
# sudo firewall-cmd --permanent --add-service=http
# sudo firewall-cmd --permanent --add-service=https
sudo firewall-cmd --reload
```

## 3. Environment Setup

### 3.1 Install Git and Docker
1. **Install Git and dependencies:**
   ```bash
   sudo dnf install -y git dnf-plugins-core
   ```

2. **Add Docker Repository:**
   ```bash
   sudo dnf config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo
   ```

3. **Install Docker Engine & Compose:**
   ```bash
   sudo dnf install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
   ```

4. **Start and Enable Docker:**
   ```bash
   sudo systemctl start docker
   sudo systemctl enable docker
   ```

5. **Allow 'deploy' user to use Docker without sudo:**
   ```bash
   sudo usermod -aG docker $USER
   # Log out and back in for this to take effect
   exit
   ssh deploy@your_server_ip
   ```

## 4. Application Deployment

### 4.1 Clone Repository
```bash
git clone https://github.com/yourusername/project_alpha.git
cd project_alpha
```

### 4.2 Configure Secrets
Create the production environment file.
```bash
cp .env.example .env
nano .env
```
**Important:**
- Set `FINNHUB_API_KEY` to your production key.
- Ensure `DATA_DIR` and `LOG_DIR` are correct (default `./data`, `./logs`).

### 4.3 Build and Run
```bash
# Build the production image
docker compose build

# Start the service in the background
docker compose up -d

# Check status
docker compose ps
```

## 5. Operations & Maintenance

### 5.1 Viewing Logs
```bash
# Follow logs in real-time
docker compose logs -f
```

### 5.2 Updating the Application
When you have pushed code changes to Git:
```bash
git pull origin main
docker compose build
docker compose up -d
```
Docker Compose will recreate only the containers that have changed.

### 5.3 Automated Tasks
The application currently runs via CLI. To schedule periodic runs (e.g., daily scan at 6 AM), use `cron`.

1. **Ensure Cron Service is Running:**
   ```bash
   sudo dnf install -y cronie
   sudo systemctl enable --now crond
   ```

2. **Edit Crontab:**
   ```bash
   crontab -e
   ```

3. **Add Job:**
   ```cron
   # Run daily US market scan at 6:00 AM UTC
   0 6 * * * cd /home/deploy/project_alpha && /usr/bin/docker compose run --rm app --market us --top 20
   ```

### 5.4 Backups
The `data/` directory contains your SQLite database and cache. Backup this directory regularly.
```bash
# Example manual backup
tar -czvf project_alpha_backup_$(date +%F).tar.gz data/
```
