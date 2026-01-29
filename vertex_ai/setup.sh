#!/bin/bash

echo "==================================================="
echo "D-FINE Vertex AI Setup Script"
echo "==================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running from D-FINE root directory
if [ ! -f "train.py" ]; then
    echo -e "${RED}Error: Please run this script from the D-FINE root directory${NC}"
    exit 1
fi

echo -e "${GREEN}✓${NC} Running from D-FINE root directory"

# Install Python dependencies for Vertex AI
echo -e "\n${YELLOW}Installing Python dependencies...${NC}"
pip install -r vertex_ai/requirements.txt

# Check Google Cloud CLI
echo -e "\n${YELLOW}Checking Google Cloud CLI...${NC}"
if command -v gcloud &> /dev/null; then
    echo -e "${GREEN}✓${NC} Google Cloud CLI found"
    CURRENT_PROJECT=$(gcloud config get-value project 2>/dev/null)
    echo "  Current project: $CURRENT_PROJECT"

    if [ "$CURRENT_PROJECT" != "production-paladium" ]; then
        echo -e "${YELLOW}  Setting project to production-paladium...${NC}"
        gcloud config set project production-paladium
    fi
else
    echo -e "${RED}✗${NC} Google Cloud CLI not found. Please install it:"
    echo "  https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Check Docker
echo -e "\n${YELLOW}Checking Docker...${NC}"
if command -v docker &> /dev/null; then
    echo -e "${GREEN}✓${NC} Docker found"
else
    echo -e "${RED}✗${NC} Docker not found. Please install Docker Desktop"
    exit 1
fi

# Authenticate with Google Cloud
echo -e "\n${YELLOW}Authenticating with Google Cloud...${NC}"
echo "Please follow the prompts to authenticate:"
gcloud auth application-default login
gcloud auth configure-docker us-central1-docker.pkg.dev

echo -e "\n==================================================="
echo -e "${GREEN}Setup complete!${NC}"
echo -e "==================================================="
echo -e "\nNext steps:"
echo "1. Upload your dataset to GCS:"
echo "   gsutil -m cp -r data/vehicle_full_coco gs://cv_models_dir/vehicle_detection/data/"
echo ""
echo "2. Build the Docker image:"
echo "   docker build -f vertex_ai/Dockerfile -t us-central1-docker.pkg.dev/production-paladium/cv-models-envs/dfine-vehicle-detection:latest --platform linux/amd64 ."
echo ""
echo "3. Push the Docker image:"
echo "   docker push us-central1-docker.pkg.dev/production-paladium/cv-models-envs/dfine-vehicle-detection:latest"
echo ""
echo "4. Submit training job:"
echo "   python vertex_ai/run_on_vertex_ai.py"
echo ""
echo -e "${YELLOW}Note: Make sure vertex_ai/vertex.yaml is configured correctly${NC}"