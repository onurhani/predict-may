"""
Create the predict_may database in MotherDuck
"""
import os
import duckdb

token = os.getenv("MOTHERDUCK_TOKEN")

if not token:
    print("❌ MOTHERDUCK_TOKEN not set. Run:")
    print("export MOTHERDUCK_TOKEN='your_token'")
    exit(1)

print("🦆 Creating predict_may database in MotherDuck...")

# Connect to MotherDuck without specifying a database
# This connects to your default database
conn = duckdb.connect(f"md:?motherduck_token={token}")

print("✓ Connected to MotherDuck")

# Create the database
try:
    conn.execute("CREATE DATABASE IF NOT EXISTS predict_may")
    print("✓ Database 'predict_may' created")
except Exception as e:
    print(f"Note: {e}")
    # Might already exist, that's fine

# Switch to it
conn.execute("USE predict_may")
print("✓ Using database 'predict_may'")

# Create schema
conn.execute("CREATE SCHEMA IF NOT EXISTS raw")
print("✓ Schema 'raw' created")

# Verify
result = conn.execute("SELECT current_database()").fetchone()
print(f"\n✅ Setup complete! Current database: {result[0]}")

# List schemas
schemas = conn.execute("""
    SELECT schema_name 
    FROM information_schema.schemata
""").fetchall()
print(f"Available schemas: {[s[0] for s in schemas]}")

conn.close()
print("\n🎉 Ready to sync data!")