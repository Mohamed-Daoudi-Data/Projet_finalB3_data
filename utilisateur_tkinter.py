import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from database import DatabaseManager
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

class MainApplication(tk.Tk):
    def __init__(self):
        super().__init__()
        self.db_manager = DatabaseManager('database_insurance_claims.db')
        self.style = ttk.Style(self)
        self.style.theme_use('clam')
        self.title("Insurance Claims Application")
        self.geometry("800x600")
        self.model = None
        self.models = {}
        self.initialize_ui()
        sns.set(style="whitegrid")

    def initialize_ui(self):
        self.configure(background='#f0f0f0')
        header_frame = ttk.Frame(self, padding="10 10 10 5")
        header_frame.pack(fill=tk.X, expand=False)

        ttk.Label(header_frame, text="Insurance Claims Visualization", font=("Helvetica", 18, "bold"), background='#f0f0f0').pack(side=tk.LEFT, padx=10)

        content_frame = ttk.Frame(self, padding="3 3 12 12")
        content_frame.pack(fill=tk.BOTH, expand=True)

        visual_button = ttk.Button(content_frame, text="Visualisation", command=self.open_visualization_window)
        visual_button.pack(pady=20)

        train_rf_button = ttk.Button(content_frame, text="Train Random Forest Model", command=self.train_random_forest)
        train_rf_button.pack(pady=10)

        train_dt_button = ttk.Button(content_frame, text="Train Decision Tree Model", command=self.train_decision_tree)
        train_dt_button.pack(pady=10)

        ttk.Label(content_frame, text="Choose Model for Prediction:").pack(pady=10)
        self.model_choice = ttk.Combobox(content_frame, values=["Random Forest", "Decision Tree"])
        self.model_choice.pack(pady=10)
        self.model_choice.set("Random Forest")

        predict_button = ttk.Button(content_frame, text="Enter Data for Prediction", command=self.open_prediction_window)
        predict_button.pack(pady=10)

    def open_visualization_window(self):
        self.visual_window = tk.Toplevel(self)
        self.visual_window.title("Data Visualizations")
        self.visual_window.geometry("800x600")
        self.visual_window.configure(background='#f0f0f0')

        ttk.Button(self.visual_window, text="Plot Data", command=self.plot_data).pack(pady=10)
        ttk.Button(self.visual_window, text="Histogram", command=self.plot_histogram).pack(pady=10)
        ttk.Button(self.visual_window, text="Scatter Plot", command=self.plot_scatter).pack(pady=10)
        ttk.Button(self.visual_window, text="Bar Chart - Claims by State", command=self.plot_bar_chart).pack(pady=10)
        ttk.Button(self.visual_window, text="Histogram - Total Claims", command=self.plot_total_claims_histogram).pack(pady=10)

    def plot_data(self):
        data = self.db_manager.fetch_some_data()
        fig, ax = plt.subplots()
        ax.plot(data['x'], data['y'])
        self.display_figure(fig)

    def plot_histogram(self):
        data = self.db_manager.fetch_histogram_data()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.histplot(data['age'], bins=30, color='blue', kde=True, ax=ax)
        ax.set_title('Distribution de l\'âge des assurés')
        ax.set_xlabel('Âge')
        ax.set_ylabel('Fréquence')
        self.display_figure(fig)

    def plot_bar_chart(self):
        data = self.db_manager.fetch_claims_by_state()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.barplot(x=data['state'], y=data['claims_count'], ax=ax)
        ax.set_title('Nombre de réclamations par état')
        ax.set_xlabel('État')
        ax.set_ylabel('Nombre de réclamations')
        self.display_figure(fig)

    def plot_total_claims_histogram(self):
        data = self.db_manager.fetch_total_claims_data()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.histplot(data['total_claim_amount'], bins=30, color='green', kde=True, ax=ax)
        ax.set_title('Distribution des réclamations totales')
        ax.set_xlabel('Montant total des réclamations')
        ax.set_ylabel('Fréquence')
        self.display_figure(fig)

    def plot_scatter(self):
        data = self.db_manager.fetch_scatter_data()
        fig, ax = plt.subplots()
        ax.scatter(data['x'], data['y'])
        self.display_figure(fig)

    def display_figure(self, fig):
        canvas = FigureCanvasTkAgg(fig, master=self.visual_window)
        canvas.draw()
        canvas.get_tk_widget().pack(expand=True)

    def train_model(self, model, model_name):
        df = pd.read_csv("insurance_claims.csv")
        df.fillna(method='ffill', inplace=True)

        label_encoders = {}
        for column in df.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column])
            label_encoders[column] = le

        selected_columns = [
            'months_as_customer', 'policy_deductable', 'umbrella_limit',
            'capital-gains', 'capital-loss', 'incident_hour_of_the_day',
            'number_of_vehicles_involved', 'bodily_injuries', 'witnesses',
            'injury_claim', 'property_claim', 'vehicle_claim'
        ]

        X = df[selected_columns]
        y = df['fraud_reported']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model.fit(X_train, y_train)
        self.models[model_name] = model

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred)

        print(f"{model_name} Accuracy: {accuracy}")
        print(f"{model_name} Confusion Matrix:")
        print(conf_matrix)
        print(f"{model_name} Classification Report:")
        print(class_report)

        messagebox.showinfo(f"{model_name} Training Complete", f"{model_name} trained successfully.")

    def train_random_forest(self):
        model = RandomForestClassifier(criterion='entropy', max_depth=10, max_features='sqrt', min_samples_leaf=1, min_samples_split=3, n_estimators=140)
        self.train_model(model, "Random Forest")

    def train_decision_tree(self):
        model = DecisionTreeClassifier(criterion='entropy', max_depth=10, min_samples_leaf=1, min_samples_split=3)
        self.train_model(model, "Decision Tree")

    def open_prediction_window(self):
        self.prediction_window = tk.Toplevel(self)
        self.prediction_window.title("Enter Data for Prediction")
        self.prediction_window.geometry("400x600")
        self.entries = {}

        fields = [
            'months_as_customer', 'policy_deductable', 'umbrella_limit',
            'capital-gains', 'capital-loss', 'incident_hour_of_the_day',
            'number_of_vehicles_involved', 'bodily_injuries', 'witnesses',
            'injury_claim', 'property_claim', 'vehicle_claim'
        ]

        for field in fields:
            row = ttk.Frame(self.prediction_window)
            row.pack(fill=tk.X, padx=5, pady=5)

            label = ttk.Label(row, text=field, width=22)
            label.pack(side=tk.LEFT)

            entry = ttk.Entry(row)
            entry.pack(side=tk.RIGHT, expand=tk.YES, fill=tk.X)
            self.entries[field] = entry

        submit_button = ttk.Button(self.prediction_window, text="Predict", command=self.predict)
        submit_button.pack(pady=10)

    def predict(self):
        model_name = self.model_choice.get()
        model = self.models.get(model_name)
        if model is None:
            messagebox.showerror("Error", f"{model_name} model is not trained yet.")
            return

        input_data = {field: float(entry.get()) for field, entry in self.entries.items()}
        input_df = pd.DataFrame([input_data])

        prediction = model.predict(input_df)
        result = "Fraud Reported" if prediction[0] == 1 else "No Fraud Reported"
        messagebox.showinfo("Prediction Result", result)
