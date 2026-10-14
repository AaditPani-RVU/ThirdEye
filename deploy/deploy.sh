#!/usr/bin/env bash
# Deploy Third Eye to Google Cloud (CS3235 requirements doc, section 11).
#
#   FastAPI on Cloud Run (asia-south1) -- serves the API, the WebSocket and the UI
#   Firestore (metadata, alerts) + Cloud Storage (spec JSON, session logs)
#   Secret Manager: GROQ_API_KEY, THIRDEYE_EDGE_TOKEN
#   Firebase Auth (Google sign-in); Firebase Hosting in front (optional)
#
# Run it from the repository root, signed in to gcloud as a project owner:
#
#   gcloud auth login
#   PROJECT=thirdeye-cs3235 ALLOWED=you@x.com,teammate@y.com ./deploy/deploy.sh
#
# Safe to run again: every step checks before it creates. It prints the
# service URL and the edge token's location at the end.
#
# One step is manual, in the Firebase console (no CLI for it): add the project
# to Firebase, enable Google as a sign-in provider, register a web app, and
# put its config in web/.env.production (see web/.env.example). Then re-run
# with FIREBASE=1 so the UI is built with sign-in.

set -euo pipefail

PROJECT="${PROJECT:?set PROJECT to the Google Cloud project id}"
REGION="${REGION:-asia-south1}"
SERVICE="${SERVICE:-thirdeye}"
BUCKET="${BUCKET:-${PROJECT}-thirdeye}"
ALLOWED="${ALLOWED:-}"
FIREBASE="${FIREBASE:-0}"
SA="${SERVICE}-run@${PROJECT}.iam.gserviceaccount.com"

say() { printf '\n== %s\n' "$*"; }
gc() { gcloud --project "$PROJECT" "$@"; }

say "APIs"
gc services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com \
  firestore.googleapis.com storage.googleapis.com secretmanager.googleapis.com \
  identitytoolkit.googleapis.com

say "Firestore (native mode, $REGION)"
gc firestore databases describe --database='(default)' >/dev/null 2>&1 \
  || gc firestore databases create --location="$REGION" --type=firestore-native

say "Bucket gs://$BUCKET"
gc storage buckets describe "gs://$BUCKET" >/dev/null 2>&1 \
  || gc storage buckets create "gs://$BUCKET" --location="$REGION" --uniform-bucket-level-access

say "Secrets"
secret() {  # name, value-or-empty: creates the secret once; never prints the value
  local name="$1" value="$2"
  if ! gc secrets describe "$name" >/dev/null 2>&1; then
    [ -n "$value" ] || { echo "  $name: no value given; set it in the environment and re-run"; return 1; }
    printf '%s' "$value" | gc secrets create "$name" --data-file=- --replication-policy=automatic
    echo "  $name created"
  else
    echo "  $name exists (add a version to rotate it)"
  fi
}
# The LLM key comes from the local .env if it is not in the environment.
if [ -z "${GROQ_API_KEY:-}" ] && [ -f .env ]; then
  GROQ_API_KEY="$(grep -E '^GROQ_API_KEY=' .env | head -1 | cut -d= -f2- | tr -d "'\"")"
fi
secret groq-api-key "${GROQ_API_KEY:-}"
secret thirdeye-edge-token "$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"

say "Service account $SA"
gc iam service-accounts describe "$SA" >/dev/null 2>&1 \
  || gc iam service-accounts create "${SERVICE}-run" --display-name="Third Eye Cloud Run"
for role in roles/datastore.user roles/secretmanager.secretAccessor; do
  gc projects add-iam-policy-binding "$PROJECT" --member="serviceAccount:$SA" --role="$role" \
    --condition=None >/dev/null
done
gcloud storage buckets add-iam-policy-binding "gs://$BUCKET" \
  --member="serviceAccount:$SA" --role=roles/storage.objectAdmin >/dev/null

say "Build args (Firebase web config)"
BUILD_ARGS=()
if [ "$FIREBASE" = "1" ]; then
  [ -f web/.env.production ] || { echo "FIREBASE=1 needs web/.env.production"; exit 1; }
  set -a; . web/.env.production; set +a
  AUTH=firebase
else
  echo "  FIREBASE=0: deploying without sign-in. Anyone with the URL can use it."
  AUTH=none
fi

say "Cloud Run service $SERVICE"
# One instance: a session lives in the server's memory, and the WebSocket
# fan-out assumes every browser talks to the same process.
IMAGE="$REGION-docker.pkg.dev/$PROJECT/thirdeye/server:latest"
gc artifacts repositories describe thirdeye --location="$REGION" >/dev/null 2>&1 \
  || gc artifacts repositories create thirdeye --repository-format=docker --location="$REGION"
gc builds submit --region="$REGION" --config=deploy/cloudbuild.yaml \
  --substitutions="_IMAGE=$IMAGE,_FB_API_KEY=${VITE_FIREBASE_API_KEY:-},_FB_AUTH_DOMAIN=${VITE_FIREBASE_AUTH_DOMAIN:-},_FB_PROJECT_ID=${VITE_FIREBASE_PROJECT_ID:-},_FB_APP_ID=${VITE_FIREBASE_APP_ID:-}" .
gc run deploy "$SERVICE" --image="$IMAGE" --region="$REGION" --service-account="$SA" \
  --allow-unauthenticated --min-instances=0 --max-instances=1 --session-affinity \
  --timeout=3600 --memory=1Gi --cpu=1 \
  --set-env-vars="THIRDEYE_STORE=firestore,THIRDEYE_BUCKET=$BUCKET,THIRDEYE_AUTH=$AUTH,FIREBASE_PROJECT=$PROJECT,THIRDEYE_ALLOWED_EMAILS=${ALLOWED//,/;}" \
  --set-secrets="GROQ_API_KEY=groq-api-key:latest,THIRDEYE_EDGE_TOKEN=thirdeye-edge-token:latest"

URL="$(gc run services describe "$SERVICE" --region="$REGION" --format='value(status.url)')"
say "Done"
echo "  URL:        $URL"
echo "  health:     curl $URL/healthz"
echo "  edge token: gcloud secrets versions access latest --secret=thirdeye-edge-token --project=$PROJECT"
echo "              put it in the demo laptop's .env as THIRDEYE_EDGE_TOKEN=..., then"
echo "              python -m thirdeye.server.edge --cloud $URL --scenario jenga_fouls"
echo
echo "  Budget alert (NFR-COST): Billing -> Budgets & alerts -> create, INR 500."
