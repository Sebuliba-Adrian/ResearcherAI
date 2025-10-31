# Terraform Variables for ResearcherAI Infrastructure

# ============================================================================
# Provider Configuration
# ============================================================================

variable "do_token" {
  description = "DigitalOcean API token"
  type        = string
  sensitive   = true
}

# ============================================================================
# Project Configuration
# ============================================================================

variable "project_name" {
  description = "Name of the project"
  type        = string
  default     = "researcherai"
}

variable "environment" {
  description = "Environment name (dev, staging, production)"
  type        = string
  default     = "production"

  validation {
    condition     = contains(["dev", "staging", "production"], var.environment)
    error_message = "Environment must be dev, staging, or production."
  }
}

variable "region" {
  description = "DigitalOcean region"
  type        = string
  default     = "nyc3"

  validation {
    condition = contains([
      "nyc1", "nyc3", "sfo1", "sfo2", "sfo3",
      "ams3", "sgp1", "lon1", "fra1", "tor1", "blr1"
    ], var.region)
    error_message = "Region must be a valid DigitalOcean region."
  }
}

# ============================================================================
# SSH Configuration
# ============================================================================

variable "ssh_key_ids" {
  description = "List of SSH key IDs for Droplet access"
  type        = list(string)
  default     = []
}

# ============================================================================
# Application Server Configuration
# ============================================================================

variable "app_server_count" {
  description = "Number of application server droplets"
  type        = number
  default     = 2

  validation {
    condition     = var.app_server_count >= 1 && var.app_server_count <= 10
    error_message = "App server count must be between 1 and 10."
  }
}

variable "app_server_size" {
  description = "Droplet size for application servers"
  type        = string
  default     = "s-2vcpu-4gb"

  # Common sizes:
  # s-1vcpu-2gb   - Basic ($18/mo)
  # s-2vcpu-4gb   - Standard ($36/mo)
  # s-4vcpu-8gb   - Performance ($72/mo)
  # s-8vcpu-16gb  - High Performance ($144/mo)
}

# ============================================================================
# Database Configuration
# ============================================================================

variable "db_size" {
  description = "Managed database size"
  type        = string
  default     = "db-s-2vcpu-4gb"

  # Common sizes:
  # db-s-1vcpu-1gb   - Basic ($15/mo)
  # db-s-2vcpu-4gb   - Standard ($60/mo)
  # db-s-4vcpu-8gb   - Performance ($120/mo)
}

variable "db_node_count" {
  description = "Number of database nodes (1 or 2 for standby)"
  type        = number
  default     = 1

  validation {
    condition     = var.db_node_count >= 1 && var.db_node_count <= 2
    error_message = "Database node count must be 1 or 2."
  }
}

# ============================================================================
# Kafka Configuration
# ============================================================================

variable "kafka_size" {
  description = "Managed Kafka cluster size"
  type        = string
  default     = "db-s-2vcpu-2gb"

  # Note: DigitalOcean Kafka requires minimum 3 nodes
}

variable "kafka_node_count" {
  description = "Number of Kafka broker nodes (minimum 3)"
  type        = number
  default     = 3

  validation {
    condition     = var.kafka_node_count >= 3
    error_message = "Kafka requires minimum 3 nodes for high availability."
  }
}

# ============================================================================
# Application Secrets (from .env or secrets manager)
# ============================================================================

variable "google_api_key" {
  description = "Google Gemini API key"
  type        = string
  sensitive   = true
}

variable "neo4j_uri" {
  description = "Neo4j connection URI (external managed service)"
  type        = string
  default     = ""
}

variable "neo4j_password" {
  description = "Neo4j password"
  type        = string
  sensitive   = true
  default     = ""
}

variable "qdrant_url" {
  description = "Qdrant Cloud URL (external managed service)"
  type        = string
  default     = ""
}

# ============================================================================
# Monitoring & Logging
# ============================================================================

variable "enable_monitoring" {
  description = "Enable DigitalOcean monitoring agent"
  type        = bool
  default     = true
}

variable "enable_backups" {
  description = "Enable automated backups for droplets"
  type        = bool
  default     = true
}

# ============================================================================
# Tags
# ============================================================================

variable "tags" {
  description = "Additional tags for resources"
  type        = list(string)
  default     = []
}
