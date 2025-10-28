#!/bin/bash
# Setup and Start Apache Airflow for ResearcherAI

set -e

echo "🚀 Setting up Apache Airflow for ResearcherAI"
echo "=============================================="

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}❌ Error: Docker is not running${NC}"
    echo "Please start Docker and try again"
    exit 1
fi

echo -e "${GREEN}✅ Docker is running${NC}"

# Check if docker-compose is installed
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo -e "${RED}❌ Error: docker-compose is not installed${NC}"
    echo "Please install docker-compose and try again"
    exit 1
fi

echo -e "${GREEN}✅ docker-compose is installed${NC}"

# Set AIRFLOW_UID to current user
export AIRFLOW_UID=$(id -u)
echo "AIRFLOW_UID=$AIRFLOW_UID" > .env.local

# Create required directories
echo -e "${YELLOW}📁 Creating required directories...${NC}"
mkdir -p ./dags ./logs ./plugins ./config
chmod -R 777 ./logs  # Airflow needs write access

# Initialize Airflow database
echo -e "${YELLOW}🗄️ Initializing Airflow database...${NC}"
docker compose up airflow-init

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Failed to initialize Airflow${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Airflow database initialized${NC}"

# Start Airflow services
echo -e "${YELLOW}🚀 Starting Airflow services...${NC}"
docker compose up -d

# Wait for services to be healthy
echo -e "${YELLOW}⏳ Waiting for services to be healthy...${NC}"
sleep 10

# Check service health
echo -e "${YELLOW}🏥 Checking service health...${NC}"

SERVICES=("airflow-webserver" "airflow-scheduler" "airflow-worker" "postgres" "redis")
ALL_HEALTHY=true

for service in "${SERVICES[@]}"; do
    if docker compose ps | grep -q "$service.*healthy\|$service.*running"; then
        echo -e "${GREEN}✅ $service is running${NC}"
    else
        echo -e "${RED}❌ $service is not healthy${NC}"
        ALL_HEALTHY=false
    fi
done

echo ""
echo "=============================================="
if [ "$ALL_HEALTHY" = true ]; then
    echo -e "${GREEN}🎉 Airflow is ready!${NC}"
    echo ""
    echo "Access the Airflow UI at: http://localhost:8080"
    echo "Username: airflow"
    echo "Password: airflow"
    echo ""
    echo "Monitor Celery workers at: http://localhost:5555 (Flower)"
    echo ""
    echo "To stop Airflow:"
    echo "  docker compose down"
    echo ""
    echo "To view logs:"
    echo "  docker compose logs -f airflow-scheduler"
    echo "  docker compose logs -f airflow-worker"
else
    echo -e "${RED}⚠️ Some services are not healthy${NC}"
    echo "Check logs with: docker compose logs"
fi
echo "=============================================="
