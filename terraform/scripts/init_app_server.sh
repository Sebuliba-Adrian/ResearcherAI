#!/bin/bash
# Initialization script for ResearcherAI application servers
# This script runs on first boot to set up the environment

set -e

echo "========================================="
echo "ResearcherAI - Application Server Setup"
echo "========================================="

# Update system
echo "Updating system packages..."
apt-get update
apt-get upgrade -y

# Install Docker
echo "Installing Docker..."
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
rm get-docker.sh

# Install Docker Compose
echo "Installing Docker Compose..."
DOCKER_COMPOSE_VERSION="2.20.0"
curl -L "https://github.com/docker/compose/releases/download/v$${DOCKER_COMPOSE_VERSION}/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

# Install other dependencies
echo "Installing dependencies..."
apt-get install -y git python3 python3-pip nginx

# Create application directory
echo "Creating application directory..."
mkdir -p /opt/researcherai
cd /opt/researcherai

# Clone repository (replace with your actual repo)
# git clone https://github.com/your-username/ResearcherAI.git .

# Create .env file from template
cat > /opt/researcherai/.env << EOL
# Application Configuration
GOOGLE_API_KEY=${google_api_key}

# Kafka Configuration
USE_KAFKA=true
KAFKA_BOOTSTRAP_SERVERS=${kafka_bootstrap_servers}

# Database Configuration (PostgreSQL for metadata)
POSTGRES_URI=${postgres_connection_uri}

# Neo4j Configuration (if using external managed service)
USE_NEO4J=${neo4j_uri != "" ? "true" : "false"}
NEO4J_URI=${neo4j_uri}
NEO4J_PASSWORD=${neo4j_password}

# Qdrant Configuration (if using external managed service)
USE_QDRANT=${qdrant_url != "" ? "true" : "false"}
QDRANT_URL=${qdrant_url}

# Environment
ENVIRONMENT=production
EOL

# Set permissions
chmod 600 /opt/researcherai/.env

# Configure Nginx reverse proxy
cat > /etc/nginx/sites-available/researcherai << 'EOL'
server {
    listen 8000;
    server_name _;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /health {
        access_log off;
        return 200 "healthy\n";
        add_header Content-Type text/plain;
    }
}
EOL

ln -sf /etc/nginx/sites-available/researcherai /etc/nginx/sites-enabled/
rm -f /etc/nginx/sites-enabled/default
nginx -t && systemctl restart nginx

# Enable services
systemctl enable docker
systemctl enable nginx

# Create systemd service for Docker Compose
cat > /etc/systemd/system/researcherai.service << 'EOL'
[Unit]
Description=ResearcherAI Application
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/opt/researcherai
ExecStart=/usr/local/bin/docker-compose up -d
ExecStop=/usr/local/bin/docker-compose down
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
EOL

systemctl daemon-reload
systemctl enable researcherai.service

echo "========================================="
echo "✅ Application server setup complete!"
echo "========================================="
echo "Next steps:"
echo "1. Clone your repository to /opt/researcherai"
echo "2. Run: docker-compose up -d"
echo "3. Check logs: docker-compose logs -f"
echo "========================================="
