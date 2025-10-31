# Droplet Module Variables

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

variable "droplet_count" {
  description = "Number of droplets to create"
  type        = number
  default     = 2
}

variable "droplet_size" {
  description = "Droplet size/slug"
  type        = string
  default     = "s-2vcpu-4gb"
}

variable "vpc_uuid" {
  description = "VPC UUID for private networking"
  type        = string
}

variable "ssh_keys" {
  description = "List of SSH key IDs"
  type        = list(string)
  default     = []
}

variable "user_data" {
  description = "User data script for droplet initialization"
  type        = string
  default     = ""
}

variable "enable_monitoring" {
  description = "Enable DigitalOcean monitoring"
  type        = bool
  default     = true
}

variable "enable_backups" {
  description = "Enable automated backups"
  type        = bool
  default     = true
}

variable "enable_persistent_volume" {
  description = "Enable persistent data volumes"
  type        = bool
  default     = false
}

variable "volume_size" {
  description = "Size of persistent volumes in GB"
  type        = number
  default     = 100
}

variable "tags" {
  description = "Additional tags"
  type        = list(string)
  default     = []
}
