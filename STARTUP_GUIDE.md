# Startup & Shutdown Guide

## Starting the Project

Run these 3 commands in order:

### Step 1: Start Docker services
```bash
cd /Users/abood/CursorProjects/capstone_CL
docker compose up -d
```
Wait ~15 seconds for all services to start.

### Step 2: Start the Cloudflare tunnel
```bash
cloudflared tunnel --url http://localhost:8000
```
This prints a URL like `https://xxxxx-xxxxx-xxxxx.trycloudflare.com`. **Copy this URL.**

### Step 3: Update the webhook URL on Meta
1. Go to https://developers.facebook.com → **Capstone_business** app
2. **WhatsApp** → **Step 2. Production setup** → **Configure Webhooks**
3. Change **Callback URL** to: `https://YOUR-NEW-URL.trycloudflare.com/webhook/whatsapp`
4. **Verify token**: `capstone_verify_2026_secret`
5. Click **"Verify and save"**

### Step 4: Update the WhatsApp token (if expired)
1. Go to https://developers.facebook.com/tools/explorer/
2. Select **Capstone_business** app
3. Click **"Generate Access Token"**
4. Copy the token
5. Update it in the **n8n workflow** "Send WhatsApp Reply" node (Authorization header)
6. Also update in the Daily Summary and Follow-up workflows if using them

---

## Shutting Down

```bash
# Stop Docker services
docker compose down

# Stop the tunnel (Ctrl+C in the tunnel terminal, or:)
pkill -f cloudflared
```

---

## What Changes Every Time

| What | When it changes | How to update |
|------|----------------|---------------|
| **Tunnel URL** | Every time you restart cloudflared | Update in Meta webhook settings (Step 3 above) |
| **WhatsApp Token** | Expires every ~1 hour | Generate new one in Graph API Explorer, update in n8n workflow nodes |
| **Nothing else** | Docker services, code, Qdrant data, n8n workflows all persist between restarts | — |

---

## Quick Health Check

After starting, verify everything works:

```bash
# Check all containers are running
docker ps

# Check backend
curl http://localhost:8000/health

# Check tunnel (use your actual URL)
curl https://YOUR-URL.trycloudflare.com/health

# Open dashboard
open http://localhost:8000/dashboard
```

Then send a WhatsApp message to +1 (555) 648-4558 to test.
