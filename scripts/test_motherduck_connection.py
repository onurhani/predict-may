"""
Test MotherDuck connection and create initial database
Run this BEFORE trying dbt
"""
import os
import duckdb
from dotenv import load_dotenv

load_dotenv()

# Get token
token = os.getenv("MOTHERDUCK_TOKEN")

if not token:
    print("❌ MOTHERDUCK_TOKEN not found in .env file")
    print("Add it like: MOTHERDUCK_TOKEN=your_token_here")
    exit(1)

print("✓ Token found")

# Connect to MotherDuck
print("\n🦆 Connecting to MotherDuck...")
try:
    # Connection string includes token
    conn = duckdb.connect(f"md:predict_may?motherduck_token={token}")
    print("✓ Connected successfully!")
    
    # Test query
    result = conn.execute("SELECT current_database()").fetchone()
    print(f"✓ Current database: {result[0]}")
    
    # Create schema if not exists
    conn.execute("CREATE SCHEMA IF NOT EXISTS raw")
    print("✓ Schema 'raw' ready")
    
    # List existing tables
    tables = conn.execute("""
        SELECT table_schema, table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'raw'
    """).fetchall()
    
    if tables:
        print(f"\n📊 Existing tables in raw schema:")
        for schema, table in tables:
            print(f"  - {schema}.{table}")
    else:
        print("\n📊 No tables in raw schema yet")
    
    conn.close()
    print("\n✅ MotherDuck connection test passed!")
    
except Exception as e:
    print(f"\n❌ Connection failed: {e}")
    print("\nTroubleshooting:")
    print("1. Check your token is correct")
    print("2. Ensure you have internet connection")
    print("3. Try running: pip install --upgrade duckdb motherduck-client")