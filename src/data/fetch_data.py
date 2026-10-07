import os
from urllib.parse import urlparse
import pg8000.native
import pandas as pd
from dotenv import load_dotenv

def get_db_connection():
    """Initializes and returns a secure, direct connection to the Supabase PostgreSQL instance."""
    load_dotenv()
    db_url = os.getenv("SUPABASE_DB_URL")
    
    if not db_url:
        raise ValueError("Database connection URL missing. Ensure SUPABASE_DB_URL is defined inside your .env file.")
    
    # Use standard library parser to perfectly isolate passwords containing special characters
    parsed_url = urlparse(db_url)
    
    user = parsed_url.username
    password = parsed_url.password
    host = parsed_url.hostname
    port = parsed_url.port or 5432
    database = parsed_url.path.lstrip('/')
    
    # Establish a native wire protocol connection
    connection = pg8000.native.Connection(
        user=user,
        password=password,
        host=host,
        port=int(port),
        database=database
    )
    return connection

def load_churn_data() -> pd.DataFrame:
    """Executes a pure SQL query against the cloud instance and parses records into a clean structural DataFrame."""
    print("Connecting securely to cloud PostgreSQL instance...")
    conn = get_db_connection()
    
    query = "SELECT * FROM public.telco_churn;"
    print(f"Executing analytical extraction query: {query}")
    
    raw_results = conn.run(query)
    columns = [col['name'] for col in conn.columns]
    conn.close()
    
    df = pd.DataFrame(raw_results, columns=columns)
    print(f"Extraction successful! Extracted shape: {df.shape}")
    return df

if __name__ == "__main__":
    try:
        data = load_churn_data()
        print("\n--- Telemetry Preview ---")
        print(data.head(2))
    except Exception as e:
        print(f"\nConnection pipeline crash encountered: {e}")
