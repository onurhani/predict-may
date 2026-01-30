"""
Sync local DuckDB data to MotherDuck
Run this after test_motherduck_connection.py succeeds
"""
import os
import duckdb
from dotenv import load_dotenv

load_dotenv()

LOCAL_DB = "data/football.duckdb"
token = os.getenv("MOTHERDUCK_TOKEN")

if not token:
    print("❌ MOTHERDUCK_TOKEN not set in environment")
    print("Run: export MOTHERDUCK_TOKEN='your_token'")
    exit(1)

if not os.path.exists(LOCAL_DB):
    print(f"❌ Local database not found: {LOCAL_DB}")
    print("Run your ingestion script first!")
    exit(1)

print("📦 Syncing local data to MotherDuck...\n")

# Connect to local DB
print("📂 Opening local database...")
local_conn = duckdb.connect(LOCAL_DB, read_only=True)

# Get row count from local
row_count = local_conn.execute("SELECT COUNT(*) FROM raw.fixtures").fetchone()[0]
print(f"✓ Local fixtures: {row_count} rows")

local_conn.close()

# Now connect to MotherDuck and copy data
print("\n🦆 Connecting to MotherDuck...")
cloud_conn = duckdb.connect(f"md:predict_may?motherduck_token={token}")

print("✓ Connected to MotherDuck")

# Create schema if needed
print("📁 Creating schema...")
cloud_conn.execute("CREATE SCHEMA IF NOT EXISTS raw")
print("✓ Schema ready")

# Attach local database to cloud connection
print("\n🔗 Attaching local database...")
cloud_conn.execute(f"ATTACH '{LOCAL_DB}' AS local_db (READ_ONLY)")
print("✓ Local database attached")

# Copy data from local to cloud
print("📊 Copying fixtures table...")
cloud_conn.execute("""
    CREATE OR REPLACE TABLE raw.fixtures AS 
    SELECT * FROM local_db.raw.fixtures
""")

# Verify
cloud_count = cloud_conn.execute("SELECT COUNT(*) FROM raw.fixtures").fetchone()[0]
print(f"✓ Cloud fixtures: {cloud_count} rows")

if row_count == cloud_count:
    print("\n✅ Sync successful!")
else:
    print(f"\n⚠️  Row count mismatch: {row_count} local vs {cloud_count} cloud")

cloud_conn.close()
print("🎉 Data sync complete!")