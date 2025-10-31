# Load Balancer Module

resource "digitalocean_loadbalancer" "main" {
  name   = "${var.project_name}-${var.environment}-lb"
  region = var.region

  # Private networking
  vpc_uuid = var.vpc_uuid

  # Forward HTTP traffic to droplets
  forwarding_rule {
    entry_protocol  = "http"
    entry_port      = 80
    target_protocol = "http"
    target_port     = 8000
  }

  # Forward HTTPS traffic (if SSL certificate configured)
  forwarding_rule {
    entry_protocol  = "https"
    entry_port      = 443
    target_protocol = "http"
    target_port     = 8000
    tls_passthrough = false
  }

  # Health check
  healthcheck {
    protocol               = "http"
    port                   = 8000
    path                   = var.health_check_path
    check_interval_seconds = 10
    response_timeout_seconds = 5
    unhealthy_threshold    = 3
    healthy_threshold      = 3
  }

  # Droplets to balance
  droplet_ids = var.droplet_ids

  # Sticky sessions (optional)
  sticky_sessions {
    type               = "cookies"
    cookie_name        = "lb"
    cookie_ttl_seconds = 3600
  }

  # Tags
  droplet_tag = "${var.project_name}-${var.environment}-app"
}
