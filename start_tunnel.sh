#!/bin/bash
# ─── Start ngrok tunnel with permanent static domain ─────────────────────────
# This URL never changes: https://scorer-upswing-amnesty.ngrok-free.dev
# Set this ONCE in Meta webhook settings and never update it again.
#
# Usage: ./start_tunnel.sh

echo "Starting ngrok tunnel..."
echo "Fixed URL: https://scorer-upswing-amnesty.ngrok-free.dev"
echo "Meta webhook URL: https://scorer-upswing-amnesty.ngrok-free.dev/webhook/whatsapp"
echo ""
echo "Press Ctrl+C to stop"
echo ""

ngrok http 8000 --url=scorer-upswing-amnesty.ngrok-free.dev
