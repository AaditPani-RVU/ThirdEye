#!/usr/bin/env bash
# Put the UI on Firebase Hosting, in front of the Cloud Run service (optional:
# Cloud Run already serves the UI at its own URL).
#
#   npm install -g firebase-tools && firebase login
#   PROJECT=thirdeye-cs3235 ./deploy/hosting.sh
#
# Hosting forwards /api/** to Cloud Run, but not WebSockets, so the build is
# pointed straight at the Cloud Run URL (VITE_API_URL) and the server is told
# to accept the Hosting origin (THIRDEYE_CORS_ORIGINS).

set -euo pipefail
PROJECT="${PROJECT:?set PROJECT}"
REGION="${REGION:-asia-south1}"
SERVICE="${SERVICE:-thirdeye}"

RUN_URL="$(gcloud run services describe "$SERVICE" --project "$PROJECT" --region "$REGION" \
  --format='value(status.url)')"
SITE="https://${PROJECT}.web.app"

[ -f web/.env.production ] || cp web/.env.example web/.env.production
grep -q '^VITE_API_URL=' web/.env.production \
  && sed -i "s|^VITE_API_URL=.*|VITE_API_URL=$RUN_URL|" web/.env.production \
  || echo "VITE_API_URL=$RUN_URL" >> web/.env.production

(cd web && npm ci --no-audit --no-fund && npm run build)
gcloud run services update "$SERVICE" --project "$PROJECT" --region "$REGION" \
  --update-env-vars="THIRDEYE_CORS_ORIGINS=$SITE;https://${PROJECT}.firebaseapp.com" >/dev/null
firebase deploy --only hosting --project "$PROJECT"
echo "UI: $SITE  (API and socket: $RUN_URL)"
echo "In Firebase console -> Authentication -> Settings -> Authorized domains, make sure"
echo "${PROJECT}.web.app and ${RUN_URL#https://} are listed."
