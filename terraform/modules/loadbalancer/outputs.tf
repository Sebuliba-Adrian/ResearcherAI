# Load Balancer Module Outputs

output "id" {
  description = "Load balancer ID"
  value       = digitalocean_loadbalancer.main.id
}

output "ip_address" {
  description = "Load balancer IP address"
  value       = digitalocean_loadbalancer.main.ip
}

output "urn" {
  description = "Load balancer URN"
  value       = digitalocean_loadbalancer.main.urn
}

output "status" {
  description = "Load balancer status"
  value       = digitalocean_loadbalancer.main.status
}
