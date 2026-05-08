#!/usr/bin/env bash
# Deploy the Kvasir classifier to Google Cloud Run.
#
# Prerequisites:
#   gcloud auth login && gcloud auth configure-docker
#   gcloud services enable run.googleapis.com containerregistry.googleapis.com
#   Train the model and confirm models/best_model.pt exists.
#
# Usage:
#   GCP_PROJECT_ID=my-project ./deploy/deploy.sh
set -euo pipefail

PROJECT_ID="${GCP_PROJECT_ID:?Set GCP_PROJECT_ID env var}"
REGION="${GCP_REGION:-us-central1}"
SERVICE_NAME="kvasir-classifier"
IMAGE="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"
MODEL_SRC="$(dirname "$0")/../models/best_model.pt"
MODEL_DEST="$(dirname "$0")/../app/model/best_model.pt"

# ── Validate ──────────────────────────────────────────────────────────────

if [[ ! -f "${MODEL_SRC}" ]]; then
  echo "ERROR: Model weights not found at ${MODEL_SRC}"
  echo "Train first: cd training && python train.py"
  exit 1
fi

# ── Copy model into app/ for Docker build context ─────────────────────────

echo "Copying model weights into app/model/ ..."
cp "${MODEL_SRC}" "${MODEL_DEST}"

# ── Build and push ────────────────────────────────────────────────────────

echo "Building Docker image: ${IMAGE}"
docker build -t "${IMAGE}" "$(dirname "$0")/../app"

echo "Pushing to Container Registry ..."
docker push "${IMAGE}"

# ── Deploy ────────────────────────────────────────────────────────────────

echo "Deploying to Cloud Run (region: ${REGION}) ..."
gcloud run deploy "${SERVICE_NAME}" \
  --image "${IMAGE}" \
  --platform managed \
  --region "${REGION}" \
  --allow-unauthenticated \
  --memory 1Gi \
  --cpu 1 \
  --min-instances 0 \
  --max-instances 3 \
  --port 8080 \
  --project "${PROJECT_ID}"

echo ""
echo "Live at:"
gcloud run services describe "${SERVICE_NAME}" \
  --region "${REGION}" \
  --project "${PROJECT_ID}" \
  --format "value(status.url)"
