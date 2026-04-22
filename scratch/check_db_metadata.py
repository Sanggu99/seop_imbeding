import os
import requests
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

url = f"{SUPABASE_URL}/rest/v1/architecture_images?select=embedding_text&limit=3"
headers = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}"
}

response = requests.get(url, headers=headers)
data = response.json()

print("--- DB Metadata Sample ---")
for i, item in enumerate(data):
    print(f"Sample {i+1}: {item['embedding_text'][:200]}...")
