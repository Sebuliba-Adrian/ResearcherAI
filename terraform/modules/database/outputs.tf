# Database Module Outputs

output "postgres_host" {
  description = "PostgreSQL host"
  value       = digitalocean_database_cluster.postgres.private_host
  sensitive   = true
}

output "postgres_port" {
  description = "PostgreSQL port"
  value       = digitalocean_database_cluster.postgres.port
}

output "postgres_database" {
  description = "PostgreSQL database name"
  value       = digitalocean_database_db.app_metadata.name
}

output "postgres_username" {
  description = "PostgreSQL username"
  value       = digitalocean_database_user.app_user.name
  sensitive   = true
}

output "postgres_password" {
  description = "PostgreSQL password"
  value       = digitalocean_database_user.app_user.password
  sensitive   = true
}

output "postgres_uri" {
  description = "PostgreSQL connection URI"
  value       = digitalocean_database_cluster.postgres.private_uri
  sensitive   = true
}

output "cluster_id" {
  description = "Database cluster ID"
  value       = digitalocean_database_cluster.postgres.id
}
