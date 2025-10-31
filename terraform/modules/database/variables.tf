# Database Module Variables

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

variable "db_size" {
  description = "Database cluster size"
  type        = string
  default     = "db-s-2vcpu-4gb"
}

variable "db_node_count" {
  description = "Number of database nodes"
  type        = number
  default     = 1
}

variable "postgres_version" {
  description = "PostgreSQL version"
  type        = string
  default     = "15"
}

variable "database_name" {
  description = "Database name"
  type        = string
  default     = "researcherai"
}

variable "database_user" {
  description = "Database username"
  type        = string
  default     = "researcherai_user"
}
