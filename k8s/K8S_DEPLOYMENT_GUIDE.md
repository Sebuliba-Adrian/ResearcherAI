# ResearcherAI Kubernetes Deployment Guide

**Version:** 2.0.0
**Status:** ✅ Production Ready
**Last Validated:** 2025-11-01
**Test Results:** 8/8 PASSED (100%)

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Prerequisites](#prerequisites)
4. [Quick Start](#quick-start)
5. [Deployment Options](#deployment-options)
6. [Testing & Validation](#testing--validation)
7. [Monitoring & Troubleshooting](#monitoring--troubleshooting)
8. [Production Checklist](#production-checklist)

---

## Overview

This directory contains a production-grade Kubernetes/Helm deployment for ResearcherAI, featuring:

- **Full Helm Chart** with 12 template files
- **Auto-scaling** with HPA (2-10 replicas)
- **High Availability** with Pod Disruption Budgets
- **Event-Driven Architecture** with Kafka (Strimzi operator)
- **Graph Database** (Neo4j StatefulSet)
- **Vector Database** (Qdrant Deployment)
- **Ingress & TLS** support
- **Comprehensive Documentation**

### Validation Results

```
✓ Chart Structure     - PASSED
✓ Chart.yaml          - PASSED
✓ values.yaml         - PASSED
✓ Template Files (12) - PASSED
✓ Resource Definitions - PASSED
✓ Security           - PASSED
✓ Documentation      - PASSED
✓ Values Structure   - PASSED
```

---

## Architecture

### Kubernetes Resources

```
researcherai/
├── Namespace (researcherai)
├── ConfigMaps & Secrets
├── Deployments
│   ├── ResearcherAI Multi-Agent (2-10 replicas, HPA)
│   ├── Qdrant (1+ replicas)
│   └── Scheduler (1 replica, optional)
├── StatefulSets
│   ├── Neo4j (1 replica, 10Gi PVC)
│   └── Kafka Cluster (3 replicas, via Strimzi)
├── Services
│   ├── researcherai (ClusterIP)
│   ├── neo4j (ClusterIP, ports 7474, 7687)
│   ├── qdrant (ClusterIP, ports 6333, 6334)
│   └── rag-kafka-bootstrap (ClusterIP, port 9092)
├── Kafka Topics (16 topics, 3 partitions each)
├── Ingress (nginx, with TLS)
├── HorizontalPodAutoscaler (CPU/Memory 80%)
└── PodDisruptionBudget (minAvailable: 1)
```

### Data Flow

```
User Request
    ↓
Ingress (HTTPS)
    ↓
ResearcherAI Service (Load Balanced)
    ↓
Multi-Agent Pods (2-10 replicas)
    ├──> Neo4j (Graph data)
    ├──> Qdrant (Vector embeddings)
    └──> Kafka (Event streaming)
```

---

## Prerequisites

### 1. Kubernetes Cluster

**Minimum Requirements:**
- Kubernetes 1.23+
- 4 CPU cores
- 16GB RAM
- 100GB storage (with dynamic provisioning)

**Recommended Cloud Providers:**
- GKE (Google Kubernetes Engine) - `gcloud container clusters create`
- EKS (Amazon Elastic Kubernetes Service) - `eksctl create cluster`
- AKS (Azure Kubernetes Service) - `az aks create`
- DOKS (DigitalOcean Kubernetes) - `doctl kubernetes cluster create`
- On-prem with k3s, kubeadm, or Rancher

### 2. Tools Required

```bash
# kubectl (Kubernetes CLI)
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
chmod +x kubectl && sudo mv kubectl /usr/local/bin/

# Helm 3.x (Package Manager)
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

# Verify installations
kubectl version --client
helm version
```

### 3. Cluster Access

```bash
# Configure kubectl context
kubectl config get-contexts
kubectl config use-context <your-cluster>

# Verify access
kubectl cluster-info
kubectl get nodes
```

---

## Quick Start

### Option A: Automated Deployment Script

```bash
# Set API key
export GOOGLE_API_KEY="your-google-api-key-here"

# Run deployment script
cd /path/to/ResearcherAI
./k8s/scripts/deploy.sh
```

The script will:
1. ✓ Check prerequisites (kubectl, helm)
2. ✓ Verify storage class
3. ✓ Install Strimzi Kafka operator
4. ✓ Configure values
5. ✓ Deploy ResearcherAI chart
6. ✓ Verify deployment status

### Option B: Manual Helm Installation

```bash
# 1. Install Strimzi Kafka Operator
kubectl create namespace kafka
helm repo add strimzi https://strimzi.io/charts/
helm install strimzi-kafka-operator strimzi/strimzi-kafka-operator \
  --namespace kafka \
  --version 0.38.0

# 2. Create custom values file
cat > custom-values.yaml <<EOF
app:
  secrets:
    googleApiKey: "YOUR_GOOGLE_API_KEY"
    neo4jPassword: "your-secure-password"

ingress:
  enabled: true
  hosts:
    - host: researcherai.yourdomain.com
      paths:
        - path: /
          pathType: Prefix
EOF

# 3. Install ResearcherAI
helm install researcherai ./k8s/helm/researcherai \
  --namespace researcherai \
  --create-namespace \
  --values custom-values.yaml

# 4. Watch deployment
kubectl get pods -n researcherai --watch
```

---

## Deployment Options

### Development/Testing (Minimal Resources)

```yaml
# dev-values.yaml
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

**Deploy:**
```bash
helm install researcherai ./k8s/helm/researcherai \
  -n researcherai --create-namespace \
  -f dev-values.yaml
```

### Production (High Availability)

```yaml
# prod-values.yaml
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

ingress:
  enabled: true
  className: nginx
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
  hosts:
    - host: researcherai.company.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: researcherai-tls
      hosts:
        - researcherai.company.com
```

**Deploy:**
```bash
helm install researcherai ./k8s/helm/researcherai \
  -n researcherai --create-namespace \
  -f prod-values.yaml \
  --set app.secrets.googleApiKey="$GOOGLE_API_KEY"
```

---

## Testing & Validation

### Pre-Deployment Validation

```bash
# Run validation tests
cd /path/to/ResearcherAI
source venv/bin/activate
python k8s/scripts/test_k8s_manifests.py
```

**Expected Output:**
```
============================================================
               Kubernetes/Helm Chart Validation
============================================================

✓ Chart Structure     - PASSED
✓ Chart.yaml          - PASSED
✓ values.yaml         - PASSED
✓ Template Files      - PASSED
✓ Resource Definitions - PASSED
✓ Security            - PASSED
✓ Documentation       - PASSED
✓ Values Structure    - PASSED

Total Tests: 8
Passed: 8
Failed: 0

✓ All tests passed! Chart is ready for deployment.
```

### Post-Deployment Verification

```bash
# Check all resources
kubectl get all -n researcherai

# Expected output:
# - 2-10 researcherai pods (Running)
# - 1 neo4j pod (Running)
# - 1 qdrant pod (Running)
# - 3 kafka pods (Running)
# - 3 zookeeper pods (Running)
# - 1 scheduler pod (Running, if enabled)

# Check Kafka cluster
kubectl get kafka -n researcherai

# Check persistent volumes
kubectl get pvc -n researcherai

# View logs
kubectl logs -l app=researcherai-multiagent -n researcherai --tail=50
```

### Access Services

```bash
# Neo4j Browser
kubectl port-forward svc/researcherai-neo4j 7474:7474 -n researcherai
# Open: http://localhost:7474
# Credentials: neo4j / research_password

# Qdrant Dashboard
kubectl port-forward svc/researcherai-qdrant 6333:6333 -n researcherai
# Open: http://localhost:6333/dashboard

# Application
kubectl port-forward svc/researcherai 8000:8000 -n researcherai
# Open: http://localhost:8000
```

---

## Monitoring & Troubleshooting

### Common Issues

**1. Pods Not Starting**

```bash
# Check pod status
kubectl describe pod <pod-name> -n researcherai

# View logs
kubectl logs <pod-name> -n researcherai

# Common causes:
# - ImagePullBackOff: Check image name/registry
# - CrashLoopBackOff: Check application logs
# - Pending: Check resource availability and PVC binding
```

**2. PVC Issues**

```bash
# Check PVC status
kubectl get pvc -n researcherai

# Describe for events
kubectl describe pvc <pvc-name> -n researcherai

# Common causes:
# - No storage class available
# - Insufficient storage quota
# - StorageClass provisioner not running
```

**3. Kafka Not Ready**

```bash
# Check Kafka cluster
kubectl get kafka -n researcherai -o yaml

# Check operator logs
kubectl logs -n kafka -l name=strimzi-cluster-operator

# Wait time: Kafka cluster can take 5-10 minutes to fully initialize
```

**4. Network Issues**

```bash
# Test connectivity between pods
kubectl exec -it <app-pod> -n researcherai -- nc -zv rag-kafka-kafka-bootstrap 9092

# Check services
kubectl get svc -n researcherai

# Check endpoints
kubectl get endpoints -n researcherai
```

### Resource Monitoring

```bash
# Pod resource usage
kubectl top pods -n researcherai

# Node resource usage
kubectl top nodes

# HPA status
kubectl get hpa -n researcherai

# Events
kubectl get events -n researcherai --sort-by='.lastTimestamp'
```

---

## Production Checklist

### Security

- [ ] Use external secret management (e.g., external-secrets operator, Vault)
- [ ] Enable network policies for pod-to-pod communication
- [ ] Configure Pod Security Standards (PSS)
- [ ] Enable TLS for Neo4j, Kafka, and Qdrant
- [ ] Implement RBAC (Role-Based Access Control)
- [ ] Regular security scanning of container images
- [ ] Rotate secrets regularly

### High Availability

- [ ] Deploy across multiple availability zones
- [ ] Configure pod anti-affinity rules
- [ ] Set appropriate resource requests/limits
- [ ] Enable HPA (Horizontal Pod Autoscaler)
- [ ] Configure PodDisruptionBudgets
- [ ] Use persistent volumes with replication

### Monitoring & Observability

- [ ] Install Prometheus for metrics
- [ ] Set up Grafana dashboards
- [ ] Configure alerting (Alertmanager, PagerDuty)
- [ ] Enable distributed tracing (Jaeger, Zipkin)
- [ ] Centralized logging (ELK stack, Loki)
- [ ] Health checks and liveness/readiness probes

### Backup & Disaster Recovery

- [ ] Schedule Neo4j backups (Velero, custom scripts)
- [ ] Back up Qdrant collections
- [ ] Document recovery procedures
- [ ] Test restore process regularly
- [ ] Store backups in separate region/zone

### Performance

- [ ] Configure appropriate resource limits
- [ ] Use fast storage class (SSD) for databases
- [ ] Enable caching where applicable
- [ ] Monitor and optimize query performance
- [ ] Regular load testing

### Compliance

- [ ] Data encryption at rest and in transit
- [ ] Audit logging enabled
- [ ] Compliance with GDPR, HIPAA (if applicable)
- [ ] Data retention policies configured

---

## Cost Estimates

### Google Cloud (GKE) - Production Config

| Resource | Quantity | Unit Cost | Monthly Cost |
|----------|----------|-----------|--------------|
| n2-standard-4 nodes | 3 | $140/mo | $420 |
| SSD Storage (200GB) | 1 | $50/mo | $50 |
| Load Balancer | 1 | $20/mo | $20 |
| **Total** | | | **$490/mo** |

### AWS (EKS) - Production Config

| Resource | Quantity | Unit Cost | Monthly Cost |
|----------|----------|-----------|--------------|
| t3.xlarge nodes | 3 | $140/mo | $420 |
| gp3 Storage (200GB) | 1 | $25/mo | $25 |
| ALB | 1 | $25/mo | $25 |
| **Total** | | | **$470/mo** |

### Azure (AKS) - Production Config

| Resource | Quantity | Unit Cost | Monthly Cost |
|----------|----------|-----------|--------------|
| Standard_D4s_v3 nodes | 3 | $155/mo | $465 |
| Premium SSD (200GB) | 1 | $55/mo | $55 |
| Load Balancer | 1 | $20/mo | $20 |
| **Total** | | | **$540/mo** |

---

## Upgrading

```bash
# Update values
vim custom-values.yaml

# Upgrade release
helm upgrade researcherai ./k8s/helm/researcherai \
  --namespace researcherai \
  --values custom-values.yaml

# Rollback if needed
helm rollback researcherai -n researcherai

# View history
helm history researcherai -n researcherai
```

---

## Uninstalling

```bash
# Uninstall the chart
helm uninstall researcherai -n researcherai

# Delete namespace (will delete all resources including PVCs)
kubectl delete namespace researcherai

# Uninstall Strimzi operator (if no longer needed)
helm uninstall strimzi-kafka-operator -n kafka
kubectl delete namespace kafka
```

---

## Support & Resources

- **GitHub Issues:** https://github.com/Sebuliba-Adrian/ResearcherAI/issues
- **Helm Chart:** `./k8s/helm/researcherai/`
- **Scripts:** `./k8s/scripts/`
- **Documentation:** `./k8s/helm/researcherai/README.md`

---

**Last Updated:** 2025-11-01
**Maintained By:** ResearcherAI Team
