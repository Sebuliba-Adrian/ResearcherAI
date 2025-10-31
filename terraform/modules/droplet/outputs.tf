# Droplet Module Outputs

output "droplet_ids" {
  description = "List of droplet IDs"
  value       = digitalocean_droplet.app[*].id
}

output "public_ips" {
  description = "List of public IP addresses"
  value       = digitalocean_droplet.app[*].ipv4_address
}

output "private_ips" {
  description = "List of private IP addresses"
  value       = digitalocean_droplet.app[*].ipv4_address_private
}

output "droplet_urns" {
  description = "List of droplet URNs"
  value       = digitalocean_droplet.app[*].urn
}

output "volume_ids" {
  description = "List of volume IDs (if enabled)"
  value       = var.enable_persistent_volume ? digitalocean_volume.data[*].id : []
}
