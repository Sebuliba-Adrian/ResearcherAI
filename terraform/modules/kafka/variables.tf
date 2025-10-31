# Kafka Module Variables

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

variable "kafka_size" {
  description = "Kafka cluster size"
  type        = string
  default     = "db-s-2vcpu-2gb"
}

variable "kafka_node_count" {
  description = "Number of Kafka nodes (minimum 3)"
  type        = number
  default     = 3
}

variable "kafka_version" {
  description = "Kafka version"
  type        = string
  default     = "3.5"
}

variable "kafka_user" {
  description = "Kafka username"
  type        = string
  default     = "kafka_app_user"
}
