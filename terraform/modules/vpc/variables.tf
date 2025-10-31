# VPC Module Variables

variable "project_name" {
  description = "Name of the project"
  type        = string
}

variable "environment" {
  description = "Environment name"
  type        = string
}

variable "vpc_region" {
  description = "DigitalOcean region for VPC"
  type        = string
}

variable "ip_range" {
  description = "IP range for VPC (CIDR notation)"
  type        = string
  default     = "10.10.0.0/16"
}
