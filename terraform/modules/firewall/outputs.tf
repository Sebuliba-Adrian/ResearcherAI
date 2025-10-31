# Firewall Module Outputs

output "firewall_id" {
  description = "Firewall ID"
  value       = digitalocean_firewall.main.id
}

output "firewall_status" {
  description = "Firewall status"
  value       = digitalocean_firewall.main.status
}
