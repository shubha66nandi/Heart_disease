import os
import sqlite3
import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

DB_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "patient_history.db")

class DatabaseManager:
    def __init__(self):
        self.use_supabase = False
        self.supabase_client = None
        
        # Try to initialize Supabase if credentials are provided
        if SUPABASE_URL and SUPABASE_KEY and SUPABASE_URL != "your_supabase_project_url" and SUPABASE_KEY != "your_supabase_anon_key":
            try:
                from supabase import create_client
                self.supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
                self.use_supabase = True
                print("Successfully connected to Supabase.")
                # We will check table existence or create it
                self._init_supabase_table()
            except Exception as e:
                print(f"Failed to connect to Supabase: {e}. Falling back to SQLite.")
                self.use_supabase = False
                
        if not self.use_supabase:
            print("Using local SQLite database.")
            self._init_sqlite_db()
            
    def _init_sqlite_db(self):
        """Initializes the local SQLite database and table if they don't exist."""
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS patient_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_name TEXT,
                age INTEGER,
                sex INTEGER,
                cp INTEGER,
                trestbps INTEGER,
                chol INTEGER,
                fbs INTEGER,
                restecg INTEGER,
                thalach INTEGER,
                exang INTEGER,
                oldpeak REAL,
                slope INTEGER,
                ca INTEGER,
                thal INTEGER,
                prediction_prob REAL,
                prediction_label INTEGER,
                created_at TEXT
            )
        """)
        conn.commit()
        conn.close()
        
    def _init_supabase_table(self):
        """
        In production with Supabase, tables are usually created in the console.
        We will print a notice or attempt a small test fetch.
        """
        try:
            # Test fetch to see if table exists
            self.supabase_client.table("patient_predictions").select("id").limit(1).execute()
        except Exception as e:
            print(f"Supabase 'patient_predictions' table not ready: {e}")
            print("Please create the 'patient_predictions' table in your Supabase dashboard.")
            print("Falling back to local SQLite database for now.")
            self.use_supabase = False
            self._init_sqlite_db()

    def save_prediction(self, patient_data, prediction_prob, prediction_label):
        """Saves a prediction entry to the active database (Supabase or SQLite)."""
        created_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Prepare payload
        record = {
            "patient_name": patient_data.get("patient_name", "Anonymous"),
            "age": int(patient_data["age"]),
            "sex": int(patient_data["sex"]),
            "cp": int(patient_data["cp"]),
            "trestbps": int(patient_data["trestbps"]),
            "chol": int(patient_data["chol"]),
            "fbs": int(patient_data["fbs"]),
            "restecg": int(patient_data["restecg"]),
            "thalach": int(patient_data["thalach"]),
            "exang": int(patient_data["exang"]),
            "oldpeak": float(patient_data["oldpeak"]),
            "slope": int(patient_data["slope"]),
            "ca": int(patient_data["ca"]),
            "thal": int(patient_data["thal"]),
            "prediction_prob": float(prediction_prob),
            "prediction_label": int(prediction_label),
            "created_at": created_at
        }
        
        if self.use_supabase:
            try:
                self.supabase_client.table("patient_predictions").insert(record).execute()
                return True, "Supabase"
            except Exception as e:
                print(f"Error saving to Supabase: {e}. Saving to SQLite local copy.")
                # Fallback save to SQLite
                self._save_to_sqlite(record)
                return True, "SQLite (Fallback)"
        else:
            self._save_to_sqlite(record)
            return True, "SQLite"

    def _save_to_sqlite(self, record):
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO patient_predictions (
                patient_name, age, sex, cp, trestbps, chol, fbs, restecg, 
                thalach, exang, oldpeak, slope, ca, thal, 
                prediction_prob, prediction_label, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            record["patient_name"], record["age"], record["sex"], record["cp"],
            record["trestbps"], record["chol"], record["fbs"], record["restecg"],
            record["thalach"], record["exang"], record["oldpeak"], record["slope"],
            record["ca"], record["thal"], record["prediction_prob"], 
            record["prediction_label"], record["created_at"]
        ))
        conn.commit()
        conn.close()

    def get_prediction_history(self, limit=None):
        """Retrieves prediction history sorted by date descending, optionally limited."""
        if self.use_supabase:
            try:
                query = self.supabase_client.table("patient_predictions").select("*").order("created_at", descending=True)
                if limit:
                    query = query.limit(limit)
                response = query.execute()
                return response.data
            except Exception as e:
                print(f"Error reading from Supabase: {e}. Reading from SQLite.")
                
        # SQLite retrieval
        conn = sqlite3.connect(DB_FILE)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        if limit:
            cursor.execute("SELECT * FROM patient_predictions ORDER BY created_at DESC LIMIT ?", (limit,))
        else:
            cursor.execute("SELECT * FROM patient_predictions ORDER BY created_at DESC")
        rows = cursor.fetchall()
        
        # Convert sqlite3.Row to list of dicts
        history = [dict(row) for row in rows]
        conn.close()
        return history

    def delete_prediction(self, record_id):
        """Deletes a prediction record by its ID."""
        if self.use_supabase:
            try:
                self.supabase_client.table("patient_predictions").delete().eq("id", record_id).execute()
                return True
            except Exception as e:
                print(f"Error deleting from Supabase: {e}")
                
        # Delete from SQLite
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM patient_predictions WHERE id = ?", (record_id,))
        conn.commit()
        conn.close()
        return True
