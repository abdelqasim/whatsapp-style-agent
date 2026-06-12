"""One-time Google Calendar OAuth setup — run this locally, not in Docker."""
from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = ["https://www.googleapis.com/auth/calendar"]

flow = InstalledAppFlow.from_client_secrets_file("./data/google_credentials.json", SCOPES)
creds = flow.run_local_server(port=8090)

with open("./data/google_token.json", "w") as f:
    f.write(creds.to_json())

print("✅ Token saved to data/google_token.json")
