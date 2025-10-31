# ResearcherAI - DigitalOcean Infrastructure

Terraform infrastructure as code for deploying ResearcherAI to DigitalOcean with production-grade architecture.

## 🏗️ Architecture

```
                                    Internet
                                        │
                                        ▼
                                 Load Balancer
                                   (Port 80/443)
                                        │
                     ┌──────────────────┼──────────────────┐
                     ▼                  ▼                  ▼
              App Server 1       App Server 2       App Server N
            (Docker Compose)   (Docker Compose)   (Docker Compose)
                     │                  │                  │
                     └──────────────────┼──────────────────┘
                                        │
                    ┌───────────────────┼───────────────────┐
                    ▼                   ▼                   ▼
            Managed Kafka        Managed PostgreSQL     External Services
          (3-node cluster)       (Metadata storage)    (Neo4j, Qdrant Cloud)
                    │                   │
                    └───────────────────┘
                            │
                       Private VPC
                     (10.10.0.0/16)
```

## 📋 Prerequisites

### 1. Install Terraform

```bash
# macOS
brew install terraform

# Linux
wget https://releases.hashicorp.com/terraform/1.6.0/terraform_1.6.0_linux_amd64.zip
unzip terraform_1.6.0_linux_amd64.zip
sudo mv terraform /usr/local/bin/

# Verify
terraform --version
```

### 2. Install DigitalOcean CLI (doctl)

```bash
# macOS
brew install doctl

# Linux
cd ~
wget https://github.com/digitalocean/doctl/releases/download/v1.98.1/doctl-1.98.1-linux-amd64.tar.gz
tar xf doctl-1.98.1-linux-amd64.tar.gz
sudo mv doctl /usr/local/bin

# Authenticate
doctl auth init
```

### 3. Get DigitalOcean API Token

1. Go to https://cloud.digitalocean.com/account/api/tokens
2. Click "Generate New Token"
3. Name: "Terraform ResearcherAI"
4. Scopes: Read & Write
5. Copy the token (you won't see it again!)

### 4. Add SSH Key to DigitalOcean

```bash
# Generate SSH key if you don't have one
ssh-keygen -t ed25519 -C "your_email@example.com"

# Add to DigitalOcean
doctl compute ssh-key create "terraform-key" --public-key "$(cat ~/.ssh/id_ed25519.pub)"

# Get SSH key ID
doctl compute ssh-key list
```

## 🚀 Quick Start

### 1. Configure Variables

```bash
cd terraform/
cp terraform.tfvars.example terraform.tfvars
nano terraform.tfvars
```

**Minimum required variables:**
- `do_token` - Your DigitalOcean API token
- `google_api_key` - Your Google Gemini API key
- `ssh_key_ids` - Your SSH key IDs

### 2. Initialize Terraform

```bash
terraform init
```

This downloads the DigitalOcean provider and sets up the working directory.

### 3. Plan Deployment

```bash
terraform plan
```

Review the resources that will be created:
- VPC (Private network)
- 2 Application server droplets
- PostgreSQL managed database
- Kafka managed cluster (3 nodes)
- Load balancer
- Firewall rules

### 4. Deploy Infrastructure

```bash
terraform apply
```

Type `yes` to confirm. Deployment takes approximately **10-15 minutes**.

### 5. Get Output Information

```bash
terraform output
```

You'll see:
- Load balancer IP address
- Application server IPs
- SSH commands
- Database connection info
- Kafka bootstrap servers

## 📊 Infrastructure Components

### VPC (Virtual Private Cloud)
- **CIDR:** 10.10.0.0/16
- **Purpose:** Private networking for secure communication
- **Cost:** Free

### Application Servers
- **Count:** 2 (configurable)
- **Size:** s-2vcpu-4gb (2 vCPU, 4GB RAM, 80GB SSD)
- **OS:** Ubuntu 22.04 LTS
- **Software:** Docker, Docker Compose, Nginx
- **Cost:** $36/mo each

### Load Balancer
- **Type:** DigitalOcean Managed Load Balancer
- **Ports:** 80 (HTTP), 443 (HTTPS)
- **Health Check:** /health endpoint
- **Cost:** $10/mo

### PostgreSQL Database
- **Size:** db-s-2vcpu-4gb (2 vCPU, 4GB RAM, 61GB SSD)
- **Version:** PostgreSQL 15
- **Nodes:** 1 (configurable to 2 for standby)
- **Purpose:** Application metadata storage
- **Cost:** $60/mo

### Kafka Cluster
- **Size:** db-s-2vcpu-2gb per node
- **Nodes:** 3 (required for high availability)
- **Version:** Kafka 3.5
- **Purpose:** Event streaming between agents
- **Cost:** $30/mo per node = $90/mo total

### Firewall
- **Inbound:** HTTP/HTTPS from load balancer, SSH from anywhere*
- **Outbound:** All traffic allowed
- **VPC:** All internal traffic allowed
- **Cost:** Free

*Note: Restrict SSH access in production!*

## 💰 Cost Breakdown

### Default Configuration

| Resource | Quantity | Unit Cost | Total |
|----------|----------|-----------|-------|
| App Servers (s-2vcpu-4gb) | 2 | $36/mo | $72/mo |
| PostgreSQL (db-s-2vcpu-4gb) | 1 | $60/mo | $60/mo |
| Kafka Cluster (db-s-2vcpu-2gb) | 3 | $30/mo | $90/mo |
| Load Balancer | 1 | $10/mo | $10/mo |
| **Total** | | | **$232/mo** |

### Cost Optimization Options

**Development/Staging:**
- App servers: s-1vcpu-2gb ($18/mo each) = $36/mo
- PostgreSQL: db-s-1vcpu-1gb ($15/mo)
- Kafka: Use Docker instead of managed = $0/mo
- **Total:** ~$61/mo

**Small Production:**
- Current default configuration
- **Total:** $232/mo

**Large Production:**
- App servers: 4x s-4vcpu-8gb ($72/mo) = $288/mo
- PostgreSQL: db-s-4vcpu-8gb ($120/mo) with standby node
- Kafka: Same (3 nodes required)
- **Total:** ~$518/mo

## 🔧 Configuration Options

### Scaling Application Servers

Edit `terraform.tfvars`:
```hcl
app_server_count = 4  # Scale from 2 to 4 servers
app_server_size = "s-4vcpu-8gb"  # Upgrade size
```

Apply changes:
```bash
terraform apply
```

### Enabling Database Standby

```hcl
db_node_count = 2  # Adds standby replica
```

### External Services

If using managed Neo4j or Qdrant Cloud:
```hcl
neo4j_uri      = "bolt://your-neo4j.cloud:7687"
neo4j_password = "your-password"
qdrant_url     = "https://your-cluster.qdrant.cloud"
```

## 📱 Deployment Steps

### After Infrastructure is Created

1. **Access Application Server:**
   ```bash
   ssh root@$(terraform output -raw app_server_ips | jq -r '.[0]')
   ```

2. **Clone Repository:**
   ```bash
   cd /opt/researcherai
   git clone https://github.com/your-username/ResearcherAI.git .
   ```

3. **Start Application:**
   ```bash
   docker-compose up -d
   ```

4. **Check Logs:**
   ```bash
   docker-compose logs -f
   ```

5. **Verify Health:**
   ```bash
   LB_IP=$(terraform output -raw loadbalancer_ip)
   curl http://$LB_IP/health
   ```

### Configure DNS

Point your domain to the load balancer IP:
```bash
# Get load balancer IP
terraform output loadbalancer_ip

# Add DNS A record
researcherai.yourdomain.com  →  <load_balancer_ip>
```

### Set Up SSL/TLS

SSH into servers and run:
```bash
apt-get install -y certbot python3-certbot-nginx
certbot --nginx -d researcherai.yourdomain.com
```

## 🔒 Security Best Practices

### 1. Restrict SSH Access

Edit `terraform.tfvars`:
```hcl
ssh_allowed_ips = ["YOUR_IP_ADDRESS/32"]
```

### 2. Use Secrets Manager

Instead of storing secrets in `terraform.tfvars`, use environment variables:
```bash
export TF_VAR_do_token="your-token"
export TF_VAR_google_api_key="your-api-key"
terraform apply
```

### 3. Enable Remote State

Store Terraform state in DigitalOcean Spaces (S3-compatible):
```hcl
terraform {
  backend "s3" {
    endpoint                    = "nyc3.digitaloceanspaces.com"
    region                      = "us-east-1"  # Ignored but required
    bucket                      = "your-bucket-name"
    key                         = "terraform.tfstate"
    skip_credentials_validation = true
    skip_metadata_api_check     = true
  }
}
```

### 4. Rotate API Keys

Regularly rotate:
- DigitalOcean API token
- Google API key
- Database passwords
- SSH keys

## 📊 Monitoring

### DigitalOcean Dashboard

View metrics at: https://cloud.digitalocean.com

- CPU usage
- Memory usage
- Disk I/O
- Network traffic

### Enable Alerts

```bash
# Create alert policy for high CPU
doctl monitoring alert policy create \
  --type v1/insights/droplet/cpu \
  --description "High CPU Usage" \
  --compare "GreaterThan" \
  --value 80 \
  --window "5m" \
  --entities $(terraform output -json droplet_ids | jq -r '.[]')
```

## 🧹 Cleanup

To destroy all infrastructure:

```bash
terraform destroy
```

**WARNING:** This will delete:
- All droplets
- Load balancer
- Databases (including data!)
- VPC
- Firewall rules

## 🐛 Troubleshooting

### Issue: "Error creating droplet"

**Cause:** Invalid SSH key ID or region not available

**Solution:**
```bash
# List available regions
doctl compute region list

# List SSH keys
doctl compute ssh-key list
```

### Issue: "Kafka cluster creation failed"

**Cause:** Kafka requires minimum 3 nodes

**Solution:** Ensure `kafka_node_count >= 3`

### Issue: "Can't connect to droplets"

**Cause:** Firewall blocking SSH

**Solution:** Check firewall rules allow your IP

### Issue: "Database connection refused"

**Cause:** Database firewall not configured

**Solution:** Verify VPC UUID matches in database firewall rule

## 📚 Additional Resources

- [DigitalOcean Terraform Provider Docs](https://registry.terraform.io/providers/digitalocean/digitalocean/latest/docs)
- [DigitalOcean API Reference](https://docs.digitalocean.com/reference/api/)
- [Terraform Best Practices](https://www.terraform-best-practices.com/)

## 🤝 Contributing

Improvements welcome! Please test in a separate environment before modifying production infrastructure.

---

**Estimated deployment time:** 10-15 minutes
**Estimated monthly cost:** $232 (default configuration)
**Scaling:** Horizontal (add more droplets) and vertical (upgrade sizes)
