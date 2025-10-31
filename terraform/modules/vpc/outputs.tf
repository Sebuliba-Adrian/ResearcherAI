# VPC Module Outputs

output "vpc_id" {
  description = "VPC UUID"
  value       = digitalocean_vpc.main.id
}

output "vpc_urn" {
  description = "VPC URN"
  value       = digitalocean_vpc.main.urn
}

output "vpc_ip_range" {
  description = "VPC IP range"
  value       = digitalocean_vpc.main.ip_range
}

output "project_id" {
  description = "Project ID"
  value       = digitalocean_project.main.id
}
