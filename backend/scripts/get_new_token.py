"""Quick script to get a new Strava refresh token"""
import requests

client_id = "198158"
client_secret = "8c83c8db8ccb8a866259e56c3d37b2b883dab1dd"

print("\n" + "="*80)
print("STRAVA TOKEN REFRESHER")
print("="*80)
print("\n1. Open this URL in your browser:\n")
print(f"https://www.strava.com/oauth/authorize?client_id={client_id}&response_type=code&redirect_uri=http://localhost&approval_prompt=force&scope=read,activity:read_all")
print("\n2. Click 'Authorize'")
print("3. Copy ONLY the code from the redirected URL")
print("   (The part after '?code=' or '&code=')")
print("\n" + "="*80 + "\n")

# Get code from user
code = input("Paste the authorization code here: ").strip()

if not code:
    print("No code provided!")
    exit(1)

# Remove any URL parts if user pasted the whole URL
if "code=" in code:
    code = code.split("code=")[1].split("&")[0]

print(f"\nExchanging code for tokens...\n")

try:
    response = requests.post(
        "https://www.strava.com/oauth/token",
        data={
            "client_id": client_id,
            "client_secret": client_secret,
            "code": code,
            "grant_type": "authorization_code",
        },
    )

    if response.status_code == 200:
        data = response.json()
        refresh_token = data["refresh_token"]

        print("="*80)
        print("SUCCESS! New refresh token:")
        print("="*80)
        print(f"\n{refresh_token}\n")
        print("="*80)
        print("\nUpdate backend/app/config/settings.py line 13 to:")
        print(f'strava_refresh_token: str = "{refresh_token}"')
        print("="*80 + "\n")
    else:
        print(f"ERROR {response.status_code}:")
        print(response.text)

except Exception as e:
    print(f"ERROR: {e}")
