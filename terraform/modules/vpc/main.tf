# VPC Module - Private Networking

resource "digitalocean_vpc" "main" {
  name        = "${var.project_name}-${var.environment}-vpc"
  region      = var.vpc_region
  description = "Private network for ${var.project_name} ${var.environment}"
  ip_range    = var.ip_range
}

# Project resource to organize resources
resource "digitalocean_project" "main" {
  name        = "${var.project_name}-${var.environment}"
  description = "ResearcherAI ${var.environment} infrastructure"
  purpose     = "Web Application"
  environment = var.environment
}
