# Custom Airflow Image with ResearcherAI Dependencies
# Based on official Apache Airflow 2.10.2 with Python 3.12

FROM apache/airflow:2.10.2-python3.12

# Switch to root to install system dependencies
USER root

# Install system dependencies if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Switch to airflow user for Python package installation
USER airflow

# Install ResearcherAI Python dependencies with compatible versions
RUN pip install --no-cache-dir \
    google-generativeai==0.8.3 \
    sentence-transformers==3.1.1 \
    faiss-cpu==1.8.0 \
    networkx==3.3 \
    duckduckgo-search==8.1.1 \
    PyPDF2==3.0.1 \
    arxiv==2.1.0 \
    pyvis==0.3.2 \
    requests==2.31.0 \
    torch==2.4.1 \
    transformers==4.46.3 \
    schedule==1.2.2 \
    rdflib==7.1.0 \
    huggingface-hub==0.27.0 \
    ipython==8.25.0 \
    jedi==0.19.1 \
    qdrant-client==1.9.1 \
    neo4j==5.23.1

# Set working directory
WORKDIR /opt/airflow

# Default command (inherited from base image)
