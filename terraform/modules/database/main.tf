# Database Module - Managed PostgreSQL

resource "digitalocean_database_cluster" "postgres" {
  name       = "${var.project_name}-${var.environment}-postgres"
  engine     = "pg"
  version    = var.postgres_version
  size       = var.db_size
  region     = var.region
  node_count = var.db_node_count

  # Private networking
  private_network_uuid = var.vpc_uuid

  tags = [
    "${var.project_name}",
    "${var.environment}",
    "postgres"
  ]
}

# Database for application metadata
resource "digitalocean_database_db" "app_metadata" {
  cluster_id = digitalocean_database_cluster.postgres.id
  name       = var.database_name
}

# Database user
resource "digitalocean_database_user" "app_user" {
  cluster_id = digitalocean_database_cluster.postgres.id
  name       = var.database_user
}

# Firewall rule to allow connections from VPC
resource "digitalocean_database_firewall" "postgres" {
  cluster_id = digitalocean_database_cluster.postgres.id

  rule {
    type  = "vpc"
    value = var.vpc_uuid
  }
}
