# Kafka Module - Managed Kafka Cluster

resource "digitalocean_database_cluster" "kafka" {
  name       = "${var.project_name}-${var.environment}-kafka"
  engine     = "kafka"
  version    = var.kafka_version
  size       = var.kafka_size
  region     = var.region
  node_count = var.kafka_node_count  # Minimum 3 for HA

  # Private networking
  private_network_uuid = var.vpc_uuid

  tags = [
    "${var.project_name}",
    "${var.environment}",
    "kafka"
  ]
}

# Kafka topics (created via API or application)
# Note: DigitalOcean Kafka auto-creates topics, but we can configure them here

# Kafka user for application
resource "digitalocean_database_user" "kafka_app_user" {
  cluster_id = digitalocean_database_cluster.kafka.id
  name       = var.kafka_user
}

# Firewall rule to allow connections from VPC
resource "digitalocean_database_firewall" "kafka" {
  cluster_id = digitalocean_database_cluster.kafka.id

  rule {
    type  = "vpc"
    value = var.vpc_uuid
  }
}
