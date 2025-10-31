# Droplet Module - Application Servers

# Data source for latest Ubuntu image
data "digitalocean_image" "ubuntu" {
  slug = "ubuntu-22-04-x64"
}

# Application server droplets
resource "digitalocean_droplet" "app" {
  count  = var.droplet_count
  name   = "${var.project_name}-${var.environment}-app-${count.index + 1}"
  region = var.region
  size   = var.droplet_size
  image  = data.digitalocean_image.ubuntu.id

  # Networking
  vpc_uuid = var.vpc_uuid
  ipv6     = true

  # SSH keys
  ssh_keys = var.ssh_keys

  # Initialization script
  user_data = var.user_data

  # Monitoring and backups
  monitoring = var.enable_monitoring
  backups    = var.enable_backups

  # Tags
  tags = concat(
    [
      "${var.project_name}",
      "${var.environment}",
      "app-server"
    ],
    var.tags
  )

  # Lifecycle
  lifecycle {
    create_before_destroy = true
  }
}

# Volume for persistent data (optional)
resource "digitalocean_volume" "data" {
  count                    = var.enable_persistent_volume ? var.droplet_count : 0
  region                   = var.region
  name                     = "${var.project_name}-${var.environment}-data-${count.index + 1}"
  size                     = var.volume_size
  initial_filesystem_type  = "ext4"
  description              = "Persistent data volume for app server ${count.index + 1}"

  tags = [
    "${var.project_name}",
    "${var.environment}",
    "data-volume"
  ]
}

# Attach volumes to droplets
resource "digitalocean_volume_attachment" "data" {
  count      = var.enable_persistent_volume ? var.droplet_count : 0
  droplet_id = digitalocean_droplet.app[count.index].id
  volume_id  = digitalocean_volume.data[count.index].id
}
