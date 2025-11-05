# ResearcherAI Helm Chart

Production-grade Kubernetes Helm chart for deploying ResearcherAI multi-agent RAG system with Neo4j, Qdrant, and Kafka.

## Overview

This Helm chart deploys a complete ResearcherAI stack including:
- **ResearcherAI Multi-Agent Application** (2+ replicas with auto-scaling)
- **Neo4j** (Graph database for knowledge graph)
- **Qdrant** (Vector database for semantic search)
- **Kafka** (Event streaming with Strimzi operator)
- **Scheduler** (Automated data collection)

## Prerequisites

### 1. Kubernetes Cluster

Minimum requirements:
- Kubernetes 1.23+
- 4 CPU cores
- 16GB RAM
- 100GB storage

Recommended cloud providers:
- Google Kubernetes Engine (GKE)
- Amazon Elastic Kubernetes Service (EKS)
- Azure Kubernetes Service (AKS)
- DigitalOcean Kubernetes (DOKS)

### 2. Helm

```bash
# Install Helm 3.x
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

# Verify installation
helm version
```

### 3. kubectl

```bash
# Install kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
chmod +x kubectl
sudo mv kubectl /usr/local/bin/

# Verify
kubectl version --client
```

### 4. Storage Class

Ensure your cluster has a default storage class or specify one in `values.yaml`:

```bash
# Check available storage classes
kubectl get storageclass

# Set default if needed
kubectl patch storageclass <name> -p '{"metadata": {"annotations":{"storageclass.kubernetes.io/is-default-class":"true"}}}'
```

## Quick Start

### 1. Add Required Helm Repositories

```bash
# Add Strimzi Kafka operator repository
helm repo add strimzi https://strimzi.io/charts/

# Update repositories
helm repo update
```

### 2. Install Strimzi Kafka Operator

```bash
# Install Strimzi operator in its own namespace
kubectl create namespace kafka
helm install strimzi-kafka-operator strimzi/strimzi-kafka-operator \
  --namespace kafka \
  --version 0.38.0
```

### 3. Install Nginx Ingress Controller (Optional)

```bash
# Install nginx ingress
helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
helm install nginx-ingress ingress-nginx/ingress-nginx \
  --create-namespace \
  --namespace ingress-nginx
```

### 4. Install cert-manager for TLS (Optional)

```bash
# Install cert-manager
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml

# Create Let's Encrypt cluster issuer
kubectl apply -f - <<EOF
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: your-email@example.com
    privateKeySecretRef:
      name: letsencrypt-prod
    solvers:
    - http01:
        ingress:
          class: nginx
EOF
```

### 5. Configure Values

Create a `custom-values.yaml` file:

```yaml
app:
  secrets:
    googleApiKey: "YOUR_GOOGLE_API_KEY"
    neo4jPassword: "YOUR_NEO4J_PASSWORD"

ingress:
  enabled: true
  hosts:
    - host: researcherai.yourdomain.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: researcherai-tls
      hosts:
        - researcherai.yourdomain.com
```

### 6. Install ResearcherAI

```bash
# Install the chart
helm install researcherai ./k8s/helm/researcherai \
  --namespace researcherai \
  --create-namespace \
  --values custom-values.yaml

# Or use --set for individual values
helm install researcherai ./k8s/helm/researcherai \
  --namespace researcherai \
  --create-namespace \
  --set app.secrets.googleApiKey="YOUR_API_KEY" \
  --set ingress.enabled=true \
  --set ingress.hosts[0].host="researcherai.example.com"
```

### 7. Verify Deployment

```bash
# Watch pods coming up
kubectl get pods -n researcherai --watch

# Check all resources
kubectl get all -n researcherai

# Check Kafka cluster
kubectl get kafka -n researcherai

# Check Neo4j
kubectl get statefulset -n researcherai

# Check services
kubectl get svc -n researcherai
```

## Configuration

### Application Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `app.replicaCount` | Number of application replicas | `2` |
| `app.image.repository` | Application image repository | `researcherai/multiagent` |
| `app.image.tag` | Application image tag | `2.0.0` |
| `app.resources.requests.cpu` | CPU request | `1000m` |
| `app.resources.requests.memory` | Memory request | `2Gi` |
| `app.resources.limits.cpu` | CPU limit | `2000m` |
| `app.resources.limits.memory` | Memory limit | `4Gi` |

### Auto-scaling Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `app.autoscaling.enabled` | Enable HPA | `true` |
| `app.autoscaling.minReplicas` | Minimum replicas | `2` |
| `app.autoscaling.maxReplicas` | Maximum replicas | `10` |
| `app.autoscaling.targetCPUUtilizationPercentage` | Target CPU % | `80` |
| `app.autoscaling.targetMemoryUtilizationPercentage` | Target Memory % | `80` |

### Neo4j Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `neo4j.enabled` | Enable Neo4j | `true` |
| `neo4j.volumes.data.defaultStorageClass.requests.storage` | Storage size | `10Gi` |
| `neo4j.config.dbms.memory.heap.max_size` | Max heap size | `2G` |

### Qdrant Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `qdrant.enabled` | Enable Qdrant | `true` |
| `qdrant.replicaCount` | Number of replicas | `1` |
| `qdrant.persistence.size` | Storage size | `10Gi` |
| `qdrant.resources.requests.cpu` | CPU request | `1000m` |
| `qdrant.resources.requests.memory` | Memory request | `2Gi` |

### Kafka Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `kafka.enabled` | Enable Kafka | `true` |
| `kafka.cluster.replicas` | Kafka broker replicas | `3` |
| `kafka.cluster.storage.size` | Storage per broker | `10Gi` |
| `kafka.zookeeper.replicas` | Zookeeper replicas | `3` |

## Deployment Scenarios

### Development/Testing

Minimal resource configuration:

```yaml
app:
  replicaCount: 1
  autoscaling:
    enabled: false
  resources:
    requests:
      cpu: 500m
      memory: 1Gi
    limits:
      cpu: 1000m
      memory: 2Gi

kafka:
  cluster:
    replicas: 1
  zookeeper:
    replicas: 1

neo4j:
  volumes:
    data:
      defaultStorageClass:
        requests:
          storage: 5Gi

qdrant:
  persistence:
    size: 5Gi
```

### Production

High-availability configuration:

```yaml
app:
  replicaCount: 3
  autoscaling:
    enabled: true
    minReplicas: 3
    maxReplicas: 20
  resources:
    requests:
      cpu: 2000m
      memory: 4Gi
    limits:
      cpu: 4000m
      memory: 8Gi

kafka:
  cluster:
    replicas: 3
    storage:
      size: 50Gi
  zookeeper:
    replicas: 3
    storage:
      size: 10Gi

neo4j:
  volumes:
    data:
      defaultStorageClass:
        requests:
          storage: 100Gi

qdrant:
  replicaCount: 3
  persistence:
    size: 100Gi
```

## Upgrading

```bash
# Update values
helm upgrade researcherai ./k8s/helm/researcherai \
  --namespace researcherai \
  --values custom-values.yaml

# Rollback if needed
helm rollback researcherai -n researcherai
```

## Uninstalling

```bash
# Uninstall the chart
helm uninstall researcherai -n researcherai

# Delete namespace (will delete all resources including PVCs)
kubectl delete namespace researcherai
```

## Monitoring

### Check Pod Status

```bash
# Get all pods
kubectl get pods -n researcherai

# Describe a pod
kubectl describe pod <pod-name> -n researcherai

# View logs
kubectl logs -f <pod-name> -n researcherai

# Logs for all replicas
kubectl logs -l app=researcherai-multiagent -n researcherai --tail=100
```

### Check Resources

```bash
# Get resource usage
kubectl top pods -n researcherai
kubectl top nodes

# Check HPA status
kubectl get hpa -n researcherai

# Check PVCs
kubectl get pvc -n researcherai
```

### Access Services

```bash
# Port forward to Neo4j browser
kubectl port-forward svc/researcherai-neo4j 7474:7474 -n researcherai
# Access at http://localhost:7474

# Port forward to Qdrant dashboard
kubectl port-forward svc/researcherai-qdrant 6333:6333 -n researcherai
# Access at http://localhost:6333/dashboard

# Port forward to application
kubectl port-forward svc/researcherai 8000:8000 -n researcherai
# Access at http://localhost:8000
```

## Troubleshooting

### Pods Not Starting

```bash
# Check pod events
kubectl describe pod <pod-name> -n researcherai

# Check logs
kubectl logs <pod-name> -n researcherai

# Common issues:
# 1. ImagePullBackOff - check image name/tag
# 2. CrashLoopBackOff - check application logs
# 3. Pending - check resource availability
```

### PVC Issues

```bash
# Check PVC status
kubectl get pvc -n researcherai

# Describe PVC for events
kubectl describe pvc <pvc-name> -n researcherai

# Check storage class
kubectl get storageclass
```

### Kafka Issues

```bash
# Check Kafka cluster status
kubectl get kafka -n researcherai

# Check Kafka pods
kubectl get pods -l strimzi.io/cluster=rag-kafka -n researcherai

# View Kafka logs
kubectl logs -l strimzi.io/name=rag-kafka-kafka -n researcherai
```

### Network Issues

```bash
# Test connectivity between pods
kubectl exec -it <pod-name> -n researcherai -- nc -zv rag-kafka-kafka-bootstrap 9092

# Check services
kubectl get svc -n researcherai

# Check endpoints
kubectl get endpoints -n researcherai
```

## Security Best Practices

1. **Use External Secret Management**: Replace built-in secrets with external-secrets operator or sealed-secrets
2. **Enable Network Policies**: Restrict pod-to-pod communication
3. **Use Pod Security Standards**: Enable PSS/PSP
4. **Regular Updates**: Keep images and dependencies updated
5. **TLS Everywhere**: Enable TLS for Kafka, Neo4j, and Qdrant
6. **RBAC**: Implement proper role-based access control

## Advanced Configuration

### Using External Databases

If you have existing Neo4j, Qdrant, or Kafka:

```yaml
neo4j:
  enabled: false
  externalUri: "bolt://external-neo4j.example.com:7687"

qdrant:
  enabled: false
  externalHost: "external-qdrant.example.com"

kafka:
  enabled: false
  externalBootstrap: "external-kafka.example.com:9092"
```

### Custom Storage Classes

```yaml
global:
  storageClass: "fast-ssd"

neo4j:
  volumes:
    data:
      defaultStorageClass:
        storageClass: "premium-ssd"
```

### Node Affinity

```yaml
affinity:
  nodeAffinity:
    requiredDuringSchedulingIgnoredDuringExecution:
      nodeSelectorTerms:
      - matchExpressions:
        - key: node-type
          operator: In
          values:
          - compute-optimized
```

## Cost Optimization

Estimated monthly costs on major cloud providers (production config):

**Google Cloud (GKE)**:
- 3x n1-standard-4 nodes: ~$360/mo
- 150GB SSD storage: ~$30/mo
- Load balancer: ~$20/mo
- **Total: ~$410/mo**

**AWS (EKS)**:
- 3x t3.xlarge nodes: ~$380/mo
- 150GB gp3 storage: ~$15/mo
- ALB: ~$25/mo
- **Total: ~$420/mo**

**Azure (AKS)**:
- 3x Standard_D4s_v3 nodes: ~$400/mo
- 150GB Premium SSD: ~$40/mo
- Load balancer: ~$20/mo
- **Total: ~$460/mo**

## Support

For issues and questions:
- GitHub Issues: https://github.com/Sebuliba-Adrian/ResearcherAI/issues
- Documentation: https://github.com/Sebuliba-Adrian/ResearcherAI

## License

See LICENSE file in the repository root.
