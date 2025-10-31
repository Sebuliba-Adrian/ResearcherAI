# Firewall Module - Security Rules

resource "digitalocean_firewall" "main" {
  name = "${var.project_name}-${var.environment}-firewall"

  # Apply to specific droplets
  droplet_ids = var.droplet_ids

  # Inbound rules

  # Allow HTTP from load balancer
  inbound_rule {
    protocol         = "tcp"
    port_range       = "80"
    source_addresses = var.loadbalancer_ip != "" ? [var.loadbalancer_ip] : ["0.0.0.0/0", "::/0"]
  }

  # Allow HTTPS from load balancer
  inbound_rule {
    protocol         = "tcp"
    port_range       = "443"
    source_addresses = var.loadbalancer_ip != "" ? [var.loadbalancer_ip] : ["0.0.0.0/0", "::/0"]
  }

  # Allow application port from load balancer
  inbound_rule {
    protocol         = "tcp"
    port_range       = "8000"
    source_addresses = var.loadbalancer_ip != "" ? [var.loadbalancer_ip] : ["0.0.0.0/0", "::/0"]
  }

  # Allow SSH from anywhere (restrict in production)
  inbound_rule {
    protocol         = "tcp"
    port_range       = "22"
    source_addresses = var.ssh_allowed_ips
  }

  # Allow internal VPC traffic
  inbound_rule {
    protocol         = "tcp"
    port_range       = "1-65535"
    source_addresses = [var.vpc_cidr]
  }

  inbound_rule {
    protocol         = "udp"
    port_range       = "1-65535"
    source_addresses = [var.vpc_cidr]
  }

  # Outbound rules

  # Allow all outbound TCP traffic
  outbound_rule {
    protocol              = "tcp"
    port_range            = "1-65535"
    destination_addresses = ["0.0.0.0/0", "::/0"]
  }

  # Allow all outbound UDP traffic
  outbound_rule {
    protocol              = "udp"
    port_range            = "1-65535"
    destination_addresses = ["0.0.0.0/0", "::/0"]
  }

  # Allow ICMP
  outbound_rule {
    protocol              = "icmp"
    destination_addresses = ["0.0.0.0/0", "::/0"]
  }
}
