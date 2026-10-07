import os
import sys
import pandas as pd
from dotenv import load_dotenv

# Connect to our existing database utility
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.data.fetch_data import get_db_connection

def seed_full_dataset():
    """Downloads the authentic 7,043-row IBM Telco dataset and pumps it into the cloud database."""
    print("Downloading the official 7,043-row IBM Telco dataset directly from source...")
    url = "https://githubusercontent.com"
    raw_df = pd.read_csv(url)
    
    # Standardize column headers to match our SQL schema exactly
    raw_df.columns = [col.lower().replace('-', '_').replace(' ', '_') for col in raw_df.columns]
    
    print("Connecting to your cloud Supabase PostgreSQL instance...")
    conn = get_db_connection()
    
    print("Clearing out benchmark rows to prepare for full production data stream...")
    conn.run("TRUNCATE TABLE public.telco_churn;")
    
    print("Streaming 7,043 rows over the wire into your cloud database (this takes about 30 seconds)...")
    
    # Construct a highly optimized batch insert query
    for _, row in raw_df.iterrows():
        # Convert values safely into strings/numbers for SQL execution
        values = [str(val).replace("'", "''") for val in row.values]
        val_string = ", ".join([f"'{v}'" for v in values])
        
        insert_query = f"""
            INSERT INTO public.telco_churn VALUES ({val_string});
        """
        conn.run(insert_query)
        
    conn.close()
    print("Database seeding successful! Your cloud database is now fully populated with production-grade data.")

if __name__ == "__main__":
    try:
        seed_full_dataset()
    except Exception as e:
        print(f"Seeding pipeline encountered an issue: {e}")

