# ResearcherAI - Production Infrastructure on DigitalOcean
# Terraform v1.0+

terraform {
  required_version = ">= 1.0"

  required_providers {
    digitalocean = {
      source  = "digitalocean/digitalocean"
      version = "~> 2.0"
    }
  }

  # Optional: Configure remote backend for state management
  # backend "s3" {
  #   bucket = "researcherai-terraform-state"
  #   key    = "production/terraform.tfstate"
  #   region = "us-east-1"
  # }
}

# DigitalOcean Provider
provider "digitalocean" {
  token = var.do_token
}

# VPC Module - Private networking
module "vpc" {
  source = "./modules/vpc"

  project_name = var.project_name
  environment  = var.environment
  vpc_region   = var.region
}

# Droplet Module - Application servers
module "droplets" {
  source = "./modules/droplet"

  project_name     = var.project_name
  environment      = var.environment
  region           = var.region
  droplet_count    = var.app_server_count
  droplet_size     = var.app_server_size
  vpc_uuid         = module.vpc.vpc_id
  ssh_keys         = var.ssh_key_ids

  # Docker image or startup script
  user_data = templatefile("${path.module}/scripts/init_app_server.sh", {
    google_api_key           = var.google_api_key
    kafka_bootstrap_servers  = module.kafka.bootstrap_servers
    postgres_connection_uri  = module.database.postgres_uri
    neo4j_uri               = var.neo4j_uri
    neo4j_password          = var.neo4j_password
    qdrant_url              = var.qdrant_url
  })

  depends_on = [module.vpc]
}

# Database Module - Managed PostgreSQL for metadata
module "database" {
  source = "./modules/database"

  project_name = var.project_name
  environment  = var.environment
  region       = var.region
  vpc_uuid     = module.vpc.vpc_id

  # PostgreSQL configuration
  db_size          = var.db_size
  db_node_count    = var.db_node_count

  depends_on = [module.vpc]
}

# Kafka Module - Managed Kafka cluster
module "kafka" {
  source = "./modules/kafka"

  project_name = var.project_name
  environment  = var.environment
  region       = var.region
  vpc_uuid     = module.vpc.vpc_id

  # Kafka configuration
  kafka_size       = var.kafka_size
  kafka_node_count = var.kafka_node_count

  depends_on = [module.vpc]
}

# Load Balancer Module - Traffic distribution
module "loadbalancer" {
  source = "./modules/loadbalancer"

  project_name  = var.project_name
  environment   = var.environment
  region        = var.region
  vpc_uuid      = module.vpc.vpc_id
  droplet_ids   = module.droplets.droplet_ids

  # Health check configuration
  health_check_path = "/health"

  depends_on = [module.droplets]
}

# Firewall Module - Security rules
module "firewall" {
  source = "./modules/firewall"

  project_name = var.project_name
  environment  = var.environment
  vpc_uuid     = module.vpc.vpc_id

  # Droplets to protect
  droplet_ids = module.droplets.droplet_ids

  # Load balancer IP
  loadbalancer_ip = module.loadbalancer.ip_address

  depends_on = [module.droplets, module.loadbalancer]
}
