import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_FOOTBALL_KEY")

headers = {
    "x-apisports-key": API_KEY
}

# Super Lig league id (API-Football)
LEAGUE_ID = 203  # Turkey Super Lig
SEASON = 2024    # örnek

resp = requests.get(
    "https://v3.football.api-sports.io/fixtures",
    headers=headers,
    params={
        "league": LEAGUE_ID,
        "season": SEASON
    }
)

data = resp.json()
print("Results:", data["results"])
print("First fixture sample:", data["response"][:1])
