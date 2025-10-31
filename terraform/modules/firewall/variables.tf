# Firewall Module Variables

variable "project_name" {
  description = "Name of the project"
  type        = string
}

variable "environment" {
  description = "Environment name"
  type        = string
}

variable "vpc_uuid" {
  description = "VPC UUID"
  type        = string
}

variable "vpc_cidr" {
  description = "VPC CIDR block"
  type        = string
  default     = "10.10.0.0/16"
}

variable "droplet_ids" {
  description = "List of droplet IDs to protect"
  type        = list(string)
}

variable "loadbalancer_ip" {
  description = "Load balancer IP address"
  type        = string
  default     = ""
}

variable "ssh_allowed_ips" {
  description = "IP addresses allowed to SSH (CIDR notation)"
  type        = list(string)
  default     = ["0.0.0.0/0", "::/0"]  # WARNING: Restrict in production!
}
