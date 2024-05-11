import sqlite3
import pandas as pd

class DatabaseManager:
    def __init__(self, db_name):
        self.db_name = db_name
        self.initialize_database()

    def connect(self):
        return sqlite3.connect(self.db_name)

    def initialize_database(self):
        self.create_insurance_claims_table()

    def create_insurance_claims_table(self):
        conn = self.connect()
        cursor = conn.cursor()
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS insurance_claims (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            months_as_customer INTEGER,
            age INTEGER,
            policy_number INTEGER,
            policy_bind_date TEXT,
            policy_state TEXT,
            policy_csl TEXT,
            policy_deductable INTEGER,
            policy_annual_premium REAL,
            umbrella_limit INTEGER,
            insured_zip INTEGER,
            insured_sex TEXT,
            insured_education_level TEXT,
            insured_occupation TEXT,
            insured_hobbies TEXT,
            insured_relationship TEXT,
            capital_gains INTEGER,
            capital_loss INTEGER,
            incident_date TEXT,
            incident_type TEXT,
            collision_type TEXT,
            incident_severity TEXT,
            authorities_contacted TEXT,
            incident_state TEXT,
            incident_city TEXT,
            incident_location TEXT,
            incident_hour_of_the_day INTEGER,
            number_of_vehicles_involved INTEGER,
            property_damage TEXT,
            bodily_injuries INTEGER,
            witnesses INTEGER,
            police_report_available TEXT,
            total_claim_amount INTEGER,
            injury_claim INTEGER,
            property_claim INTEGER,
            vehicle_claim INTEGER,
            auto_make TEXT,
            auto_model TEXT,
            auto_year INTEGER,
            fraud_reported TEXT
        );''')
        conn.commit()
        conn.close()

    def load_data_from_csv(self, csv_file):
        data = pd.read_csv(csv_file)
        data.columns = [col.replace('-', '_') for col in data.columns]
        if '_c39' in data.columns:
            data = data.drop(columns=['_c39'])
        conn = self.connect()
        data.to_sql('insurance_claims', conn, if_exists='append', index=False)
        conn.close()

    def fetch_some_data(self):
        """Fetch data for line plot visualization."""
        conn = self.connect()
        query = "SELECT incident_date, total_claim_amount FROM insurance_claims ORDER BY incident_date"
        data = pd.read_sql(query, conn)
        conn.close()
        return data

    def fetch_histogram_data(self):
        """Fetch data for histogram visualization."""
        conn = self.connect()
        query = "SELECT age FROM insurance_claims"
        data = pd.read_sql(query, conn)
        conn.close()
        return data
    
    def fetch_claims_by_state(self):
        # Connexion à la base de données et exécution d'une requête SQL
        conn = self.connect()
        query = "SELECT policy_state AS state, COUNT(*) AS claims_count FROM insurance_claims GROUP BY policy_state"
        data = pd.read_sql(query, conn)
        conn.close()
        return data

    def fetch_total_claims_data(self):
        # Connexion à la base de données et exécution d'une requête SQL
        conn = self.connect()
        query = "SELECT total_claim_amount FROM insurance_claims"
        data = pd.read_sql(query, conn)
        conn.close()
        return data

    def fetch_scatter_data(self):
        """Fetch data for scatter plot visualization."""
        conn = self.connect()
        query = "SELECT age, total_claim_amount FROM insurance_claims"
        data = pd.read_sql(query, conn)
        conn.close()
        return {'x': data['age'], 'y': data['total_claim_amount']}


db_manager = DatabaseManager('database_insurance_claims.db')
db_manager.create_insurance_claims_table()  # Create the table if not exists
db_manager.load_data_from_csv('insurance_claims.csv')  # Load data
