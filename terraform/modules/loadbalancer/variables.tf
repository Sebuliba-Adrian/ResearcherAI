# Load Balancer Module Variables

variable "project_name" {
  description = "Name of the project"
  type        = string
}

variable "environment" {
  description = "Environment name"
  type        = string
}

variable "region" {
  description = "DigitalOcean region"
  type        = string
}

variable "vpc_uuid" {
  description = "VPC UUID"
  type        = string
}

variable "droplet_ids" {
  description = "List of droplet IDs to balance"
  type        = list(string)
}

variable "health_check_path" {
  description = "Path for health check"
  type        = string
  default     = "/health"
}
