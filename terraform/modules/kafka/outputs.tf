# Kafka Module Outputs

output "bootstrap_servers" {
  description = "Kafka bootstrap servers (private network)"
  value       = "${digitalocean_database_cluster.kafka.private_host}:${digitalocean_database_cluster.kafka.port}"
}

output "kafka_host" {
  description = "Kafka private host"
  value       = digitalocean_database_cluster.kafka.private_host
  sensitive   = true
}

output "kafka_port" {
  description = "Kafka port"
  value       = digitalocean_database_cluster.kafka.port
}

output "kafka_username" {
  description = "Kafka username"
  value       = digitalocean_database_user.kafka_app_user.name
  sensitive   = true
}

output "kafka_password" {
  description = "Kafka password"
  value       = digitalocean_database_user.kafka_app_user.password
  sensitive   = true
}

output "cluster_id" {
  description = "Kafka cluster ID"
  value       = digitalocean_database_cluster.kafka.id
}
