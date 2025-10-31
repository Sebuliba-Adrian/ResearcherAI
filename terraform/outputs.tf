# Terraform Outputs for ResearcherAI Infrastructure

# ============================================================================
# VPC Outputs
# ============================================================================

output "vpc_id" {
  description = "VPC UUID"
  value       = module.vpc.vpc_id
}

output "vpc_urn" {
  description = "VPC URN"
  value       = module.vpc.vpc_urn
}

# ============================================================================
# Application Server Outputs
# ============================================================================

output "app_server_ips" {
  description = "Public IP addresses of application servers"
  value       = module.droplets.public_ips
}

output "app_server_private_ips" {
  description = "Private IP addresses of application servers"
  value       = module.droplets.private_ips
}

output "app_server_ids" {
  description = "Droplet IDs of application servers"
  value       = module.droplets.droplet_ids
}

# ============================================================================
# Load Balancer Outputs
# ============================================================================

output "loadbalancer_ip" {
  description = "Load balancer public IP address"
  value       = module.loadbalancer.ip_address
}

output "loadbalancer_url" {
  description = "Load balancer URL"
  value       = "http://${module.loadbalancer.ip_address}"
}

output "loadbalancer_id" {
  description = "Load balancer ID"
  value       = module.loadbalancer.id
}

# ============================================================================
# Database Outputs
# ============================================================================

output "postgres_host" {
  description = "PostgreSQL database host"
  value       = module.database.postgres_host
  sensitive   = true
}

output "postgres_port" {
  description = "PostgreSQL database port"
  value       = module.database.postgres_port
}

output "postgres_database" {
  description = "PostgreSQL database name"
  value       = module.database.postgres_database
}

output "postgres_username" {
  description = "PostgreSQL database username"
  value       = module.database.postgres_username
  sensitive   = true
}

output "postgres_connection_uri" {
  description = "PostgreSQL connection URI"
  value       = module.database.postgres_uri
  sensitive   = true
}

# ============================================================================
# Kafka Outputs
# ============================================================================

output "kafka_bootstrap_servers" {
  description = "Kafka bootstrap servers"
  value       = module.kafka.bootstrap_servers
}

output "kafka_cluster_id" {
  description = "Kafka cluster ID"
  value       = module.kafka.cluster_id
}

# ============================================================================
# SSH Access
# ============================================================================

output "ssh_commands" {
  description = "SSH commands to access droplets"
  value = formatlist(
    "ssh root@%s",
    module.droplets.public_ips
  )
}

# ============================================================================
# Deployment Information
# ============================================================================

output "deployment_info" {
  description = "Complete deployment information"
  value = {
    environment         = var.environment
    region             = var.region
    app_servers        = var.app_server_count
    loadbalancer       = module.loadbalancer.ip_address
    vpc_id             = module.vpc.vpc_id
    monitoring_enabled = var.enable_monitoring
    backups_enabled    = var.enable_backups
  }
}

# ============================================================================
# Cost Estimation (approximate monthly)
# ============================================================================

output "estimated_monthly_cost" {
  description = "Estimated monthly cost in USD"
  value = format("$%.2f", (
    (var.app_server_count * 36) +  # App servers (s-2vcpu-4gb)
    60 +                            # PostgreSQL (db-s-2vcpu-4gb)
    90 +                            # Kafka 3-node cluster
    10                              # Load balancer
  ))
}

# ============================================================================
# Next Steps
# ============================================================================

output "next_steps" {
  description = "Next steps after deployment"
  value = <<-EOT

    ====================================================================
    🎉 ResearcherAI Infrastructure Deployed Successfully!
    ====================================================================

    Load Balancer: http://${module.loadbalancer.ip_address}
    Environment: ${var.environment}
    Region: ${var.region}

    📋 Next Steps:

    1. Configure DNS:
       - Point your domain to: ${module.loadbalancer.ip_address}
       - Add A record: researcherai.yourdomain.com

    2. Set up SSL/TLS:
       - Install certbot on droplets
       - Run: certbot --nginx -d researcherai.yourdomain.com

    3. Verify Services:
       - Health check: http://${module.loadbalancer.ip_address}/health
       - Kafka UI: http://${module.loadbalancer.ip_address}:8081
       - Neo4j: http://${module.loadbalancer.ip_address}:7474

    4. Monitor Infrastructure:
       - DigitalOcean Dashboard: https://cloud.digitalocean.com
       - Enable alerts for droplet CPU/memory

    5. Deploy Application:
       ssh root@${module.droplets.public_ips[0]}
       cd /opt/researcherai
       docker-compose up -d

    6. Security:
       - Update firewall rules if needed
       - Rotate API keys regularly
       - Enable database backups

    ====================================================================

    Estimated Monthly Cost: $${format("%.2f", (
      (var.app_server_count * 36) + 60 + 90 + 10
    ))}

    ====================================================================
  EOT
}
