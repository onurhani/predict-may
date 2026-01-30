"""
Debug token loading and MotherDuck connection
"""
import os
from dotenv import load_dotenv

# Explicitly load .env from current directory
env_path = os.path.join(os.getcwd(), '.env')
print(f"Looking for .env at: {env_path}")
print(f".env exists: {os.path.exists(env_path)}")

if os.path.exists(env_path):
    print(f"\n.env contents (first 200 chars):")
    with open(env_path, 'r') as f:
        content = f.read()
        print(content[:200])
        print("...")

print("\n" + "="*50)
load_dotenv(env_path, override=True)

token = os.getenv("MOTHERDUCK_TOKEN")

if token:
    print(f"✅ Token loaded successfully")
    print(f"Token length: {len(token)}")
    print(f"Token starts with: {token[:10]}...")
    print(f"Token ends with: ...{token[-10:]}")
else:
    print("❌ Token NOT found in environment")
    print("\nAll environment variables starting with 'MOTHER':")
    for key, value in os.environ.items():
        if key.startswith('MOTHER'):
            print(f"  {key} = {value[:20]}...")

print("\n" + "="*50)
print("\nAttempting connection with explicit token...")

import duckdb

try:
    # Method 1: Token in connection string
    conn_string = f"md:predict_may?motherduck_token={token}"
    print(f"Connection string: md:predict_may?motherduck_token={'*' * 20}...")
    
    conn = duckdb.connect(conn_string)
    result = conn.execute("SELECT current_database()").fetchone()
    print(f"✅ Connected! Database: {result[0]}")
    conn.close()
    
except Exception as e:
    print(f"❌ Method 1 failed: {e}")
    
    # Method 2: Set environment variable before connecting
    print("\nTrying method 2: Setting motherduck_token env var...")
    os.environ['motherduck_token'] = token
    
    try:
        conn = duckdb.connect("md:predict_may")
        result = conn.execute("SELECT current_database()").fetchone()
        print(f"✅ Connected! Database: {result[0]}")
        conn.close()
    except Exception as e2:
        print(f"❌ Method 2 also failed: {e2}")
        print("\n🔍 Debugging hints:")
        print("1. Check if token is valid (regenerate from motherduck.com)")
        print("2. Ensure no spaces/quotes around token in .env")
        print("3. Try: export MOTHERDUCK_TOKEN='your_token' in terminal")
        print("4. Check internet connection")