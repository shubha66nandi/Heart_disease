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
                self._init_supabase_table()
            except Exception as e:
                print(f"Failed to connect to Supabase: {e}. Falling back to SQLite.")
                self.use_supabase = False
                
        if not self.use_supabase:
            print("Using local SQLite database.")
            self._init_sqlite_db()
            
    def _init_sqlite_db(self):
        """Initializes the local SQLite database and migrates schema if old table columns exist."""
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        # Check existing table columns if table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='patient_predictions'")
        table_exists = cursor.fetchone()
        
        if table_exists:
            cursor.execute("PRAGMA table_info(patient_predictions)")
            cols = [row[1] for row in cursor.fetchall()]
            if "male" not in cols:
                print("Detected legacy table schema. Migrating SQLite table to Framingham schema...")
                cursor.execute("DROP TABLE patient_predictions")
                conn.commit()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS patient_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_name TEXT,
                male INTEGER,
                age INTEGER,
                education INTEGER,
                currentSmoker INTEGER,
                cigsPerDay INTEGER,
                BPMeds INTEGER,
                prevalentStroke INTEGER,
                prevalentHyp INTEGER,
                diabetes INTEGER,
                totChol REAL,
                sysBP REAL,
                diaBP REAL,
                BMI REAL,
                heartRate REAL,
                glucose REAL,
                prediction_prob REAL,
                prediction_label INTEGER,
                created_at TEXT
            )
        """)
        conn.commit()
        conn.close()
        
    def _init_supabase_table(self):
        """Checks Supabase connectivity."""
        try:
            self.supabase_client.table("patient_predictions").select("id").limit(1).execute()
        except Exception as e:
            print(f"Supabase 'patient_predictions' table not ready: {e}")
            print("Falling back to local SQLite database for now.")
            self.use_supabase = False
            self._init_sqlite_db()

    def save_prediction(self, patient_data, prediction_prob, prediction_label):
        """Saves a Framingham prediction entry to the active database (Supabase or SQLite)."""
        created_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        record = {
            "patient_name": patient_data.get("patient_name", "Anonymous"),
            "male": int(patient_data.get("male", 0)),
            "age": int(patient_data.get("age", 50)),
            "education": int(patient_data.get("education", 1)),
            "currentSmoker": int(patient_data.get("currentSmoker", 0)),
            "cigsPerDay": int(patient_data.get("cigsPerDay", 0)),
            "BPMeds": int(patient_data.get("BPMeds", 0)),
            "prevalentStroke": int(patient_data.get("prevalentStroke", 0)),
            "prevalentHyp": int(patient_data.get("prevalentHyp", 0)),
            "diabetes": int(patient_data.get("diabetes", 0)),
            "totChol": float(patient_data.get("totChol", 200.0)),
            "sysBP": float(patient_data.get("sysBP", 120.0)),
            "diaBP": float(patient_data.get("diaBP", 80.0)),
            "BMI": float(patient_data.get("BMI", 25.0)),
            "heartRate": float(patient_data.get("heartRate", 75.0)),
            "glucose": float(patient_data.get("glucose", 85.0)),
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
                patient_name, male, age, education, currentSmoker, cigsPerDay, 
                BPMeds, prevalentStroke, prevalentHyp, diabetes, totChol, 
                sysBP, diaBP, BMI, heartRate, glucose, 
                prediction_prob, prediction_label, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            record["patient_name"], record["male"], record["age"], record["education"],
            record["currentSmoker"], record["cigsPerDay"], record["BPMeds"], record["prevalentStroke"],
            record["prevalentHyp"], record["diabetes"], record["totChol"], record["sysBP"],
            record["diaBP"], record["BMI"], record["heartRate"], record["glucose"],
            record["prediction_prob"], record["prediction_label"], record["created_at"]
        ))
        conn.commit()
        conn.close()

    def get_prediction_history(self, limit=None):
        """Retrieves prediction history sorted by date descending."""
        if self.use_supabase:
            try:
                query = self.supabase_client.table("patient_predictions").select("*").order("created_at", descending=True)
                if limit:
                    query = query.limit(limit)
                response = query.execute()
                return response.data
            except Exception as e:
                print(f"Error reading from Supabase: {e}. Reading from SQLite.")
                
        conn = sqlite3.connect(DB_FILE)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        if limit:
            cursor.execute("SELECT * FROM patient_predictions ORDER BY created_at DESC LIMIT ?", (limit,))
        else:
            cursor.execute("SELECT * FROM patient_predictions ORDER BY created_at DESC")
        rows = cursor.fetchall()
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
                
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM patient_predictions WHERE id = ?", (record_id,))
        conn.commit()
        conn.close()
        return True
