#!/usr/bin/env bash
#
# ResearcherAI Kubernetes Deployment Script
# Deploys ResearcherAI to a Kubernetes cluster using Helm
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
NAMESPACE="${NAMESPACE:-researcherai}"
RELEASE_NAME="${RELEASE_NAME:-researcherai}"
HELM_CHART_PATH="./k8s/helm/researcherai"
KAFKA_NAMESPACE="${KAFKA_NAMESPACE:-kafka}"

# Functions
print_header() {
    echo -e "\n${GREEN}========================================${NC}"
    echo -e "${GREEN}  $1${NC}"
    echo -e "${GREEN}========================================${NC}\n"
}

print_info() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

check_prerequisites() {
    print_header "Checking Prerequisites"

    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        print_error "kubectl not found. Please install kubectl."
        exit 1
    fi
    print_info "kubectl: $(kubectl version --client --short 2>/dev/null | head -n1)"

    # Check helm
    if ! command -v helm &> /dev/null; then
        print_error "helm not found. Please install Helm 3.x."
        exit 1
    fi
    print_info "helm: $(helm version --short)"

    # Check cluster connection
    if ! kubectl cluster-info &> /dev/null; then
        print_error "Cannot connect to Kubernetes cluster."
        print_error "Please configure kubectl with 'kubectl config use-context <context-name>'"
        exit 1
    fi
    print_info "Connected to cluster: $(kubectl config current-context)"

    # Check cluster nodes
    NODE_COUNT=$(kubectl get nodes --no-headers 2>/dev/null | wc -l)
    print_info "Cluster nodes: $NODE_COUNT"

    if [ "$NODE_COUNT" -lt 1 ]; then
        print_error "No nodes found in cluster!"
        exit 1
    fi
}

check_storage_class() {
    print_header "Checking Storage Class"

    if kubectl get storageclass &> /dev/null; then
        DEFAULT_SC=$(kubectl get storageclass -o jsonpath='{.items[?(@.metadata.annotations.storageclass\.kubernetes\.io/is-default-class=="true")].metadata.name}')

        if [ -z "$DEFAULT_SC" ]; then
            print_warning "No default storage class found."
            echo "Available storage classes:"
            kubectl get storageclass -o custom-columns=NAME:.metadata.name,PROVISIONER:.provisioner

            read -p "Enter storage class name to use (or press Enter to skip): " SC_INPUT
            if [ -n "$SC_INPUT" ]; then
                export STORAGE_CLASS="$SC_INPUT"
                print_info "Will use storage class: $STORAGE_CLASS"
            fi
        else
            print_info "Default storage class: $DEFAULT_SC"
        fi
    else
        print_warning "Unable to check storage classes"
    fi
}

install_strimzi_operator() {
    print_header "Installing Strimzi Kafka Operator"

    # Check if Strimzi is already installed
    if helm list -n "$KAFKA_NAMESPACE" | grep -q "strimzi-kafka-operator"; then
        print_info "Strimzi operator already installed"
        return
    fi

    # Add Strimzi repository
    print_info "Adding Strimzi Helm repository..."
    helm repo add strimzi https://strimzi.io/charts/ 2>/dev/null || true
    helm repo update

    # Create namespace
    kubectl create namespace "$KAFKA_NAMESPACE" 2>/dev/null || print_info "Namespace $KAFKA_NAMESPACE already exists"

    # Install operator
    print_info "Installing Strimzi Kafka operator..."
    helm install strimzi-kafka-operator strimzi/strimzi-kafka-operator \
        --namespace "$KAFKA_NAMESPACE" \
        --version 0.38.0 \
        --wait \
        --timeout 5m

    print_info "Strimzi operator installed successfully"
}

configure_values() {
    print_header "Configuring Installation Values"

    # Check for custom values file
    if [ -f "custom-values.yaml" ]; then
        print_info "Found custom-values.yaml"
        CUSTOM_VALUES="-f custom-values.yaml"
    else
        print_warning "No custom-values.yaml found"
        CUSTOM_VALUES=""
    fi

    # Check for required secrets
    if [ -z "$GOOGLE_API_KEY" ]; then
        print_warning "GOOGLE_API_KEY environment variable not set"
        read -sp "Enter Google API Key (or press Enter to skip): " API_KEY_INPUT
        echo
        if [ -n "$API_KEY_INPUT" ]; then
            export GOOGLE_API_KEY="$API_KEY_INPUT"
        fi
    fi

    # Build helm set arguments
    HELM_SETS=""
    if [ -n "$GOOGLE_API_KEY" ]; then
        HELM_SETS="$HELM_SETS --set app.secrets.googleApiKey=$GOOGLE_API_KEY"
    fi
    if [ -n "$STORAGE_CLASS" ]; then
        HELM_SETS="$HELM_SETS --set global.storageClass=$STORAGE_CLASS"
    fi

    print_info "Configuration ready"
}

install_researcherai() {
    print_header "Installing ResearcherAI"

    # Create namespace
    kubectl create namespace "$NAMESPACE" 2>/dev/null || print_info "Namespace $NAMESPACE already exists"

    # Check if release exists
    if helm list -n "$NAMESPACE" | grep -q "$RELEASE_NAME"; then
        print_info "Release $RELEASE_NAME already exists, upgrading..."
        HELM_COMMAND="upgrade"
    else
        print_info "Installing new release..."
        HELM_COMMAND="install"
    fi

    # Install/Upgrade chart
    print_info "Running: helm $HELM_COMMAND $RELEASE_NAME $HELM_CHART_PATH"

    helm $HELM_COMMAND "$RELEASE_NAME" "$HELM_CHART_PATH" \
        --namespace "$NAMESPACE" \
        --create-namespace \
        $CUSTOM_VALUES \
        $HELM_SETS \
        --wait \
        --timeout 10m

    print_info "ResearcherAI installed successfully!"
}

verify_deployment() {
    print_header "Verifying Deployment"

    print_info "Waiting for pods to be ready..."
    sleep 10

    # Check Neo4j
    NEO4J_READY=$(kubectl get statefulset -n "$NAMESPACE" -l app=neo4j -o jsonpath='{.items[0].status.readyReplicas}' 2>/dev/null || echo "0")
    if [ "$NEO4J_READY" -ge 1 ]; then
        print_info "Neo4j: Ready ($NEO4J_READY/1 replicas)"
    else
        print_warning "Neo4j: Not ready yet"
    fi

    # Check Qdrant
    QDRANT_READY=$(kubectl get deployment -n "$NAMESPACE" -l app=qdrant -o jsonpath='{.items[0].status.readyReplicas}' 2>/dev/null || echo "0")
    if [ "$QDRANT_READY" -ge 1 ]; then
        print_info "Qdrant: Ready ($QDRANT_READY replicas)"
    else
        print_warning "Qdrant: Not ready yet"
    fi

    # Check Kafka
    KAFKA_READY=$(kubectl get kafka -n "$NAMESPACE" -o jsonpath='{.items[0].status.conditions[?(@.type=="Ready")].status}' 2>/dev/null || echo "False")
    if [ "$KAFKA_READY" == "True" ]; then
        print_info "Kafka: Ready"
    else
        print_warning "Kafka: Not ready yet (this can take 5-10 minutes)"
    fi

    # Check Application
    APP_READY=$(kubectl get deployment -n "$NAMESPACE" -l app=researcherai-multiagent -o jsonpath='{.items[0].status.readyReplicas}' 2>/dev/null || echo "0")
    APP_DESIRED=$(kubectl get deployment -n "$NAMESPACE" -l app=researcherai-multiagent -o jsonpath='{.items[0].status.replicas}' 2>/dev/null || echo "0")
    if [ "$APP_READY" -ge 1 ]; then
        print_info "Application: Ready ($APP_READY/$APP_DESIRED replicas)"
    else
        print_warning "Application: Not ready yet"
    fi

    echo ""
    print_info "Deployment verification complete"
    echo ""
    echo "To check pod status:"
    echo "  kubectl get pods -n $NAMESPACE"
    echo ""
    echo "To view logs:"
    echo "  kubectl logs -l app=researcherai-multiagent -n $NAMESPACE --tail=100 -f"
    echo ""
    echo "To access Neo4j browser:"
    echo "  kubectl port-forward svc/$RELEASE_NAME-neo4j 7474:7474 -n $NAMESPACE"
    echo "  Open: http://localhost:7474"
    echo ""
    echo "To access Qdrant dashboard:"
    echo "  kubectl port-forward svc/$RELEASE_NAME-qdrant 6333:6333 -n $NAMESPACE"
    echo "  Open: http://localhost:6333/dashboard"
}

show_summary() {
    print_header "Deployment Summary"

    echo "Namespace:     $NAMESPACE"
    echo "Release Name:  $RELEASE_NAME"
    echo "Chart Path:    $HELM_CHART_PATH"
    echo ""

    print_info "All components deployed successfully!"
    echo ""
    echo "Next steps:"
    echo "1. Monitor deployment: kubectl get pods -n $NAMESPACE --watch"
    echo "2. Check logs: kubectl logs -f <pod-name> -n $NAMESPACE"
    echo "3. Access services using port-forward (see verification output above)"
    echo ""
}

# Main execution
main() {
    echo ""
    echo "╔═══════════════════════════════════════════════════════╗"
    echo "║                                                       ║"
    echo "║       ResearcherAI Kubernetes Deployment             ║"
    echo "║                                                       ║"
    echo "╚═══════════════════════════════════════════════════════╝"
    echo ""

    check_prerequisites
    check_storage_class
    install_strimzi_operator
    configure_values
    install_researcherai
    verify_deployment
    show_summary
}

# Run main function
main "$@"
