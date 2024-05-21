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

        visual_button = ttk.Button(content_frame, text="Visualisation", command=self.open_visualization_type_window)
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

    def open_visualization_type_window(self):
        self.visual_type_window = tk.Toplevel(self)
        self.visual_type_window.title("Choose Visualization Type")
        self.visual_type_window.geometry("300x200")
        self.visual_type_window.configure(background='#f0f0f0')

        general_visual_button = ttk.Button(self.visual_type_window, text="Visualisation générale", command=self.open_general_visualization_window)
        general_visual_button.pack(pady=10)

        fraud_visual_button = ttk.Button(self.visual_type_window, text="Visualisation en rapport avec les fraudes", command=self.open_fraud_visualization_window)
        fraud_visual_button.pack(pady=10)

    def open_general_visualization_window(self):
        self.general_visual_window = tk.Toplevel(self)
        self.general_visual_window.title("General Data Visualizations")
        self.general_visual_window.geometry("300x400")
        self.general_visual_window.configure(background='#f0f0f0')

        ttk.Button(self.general_visual_window, text="Distribution de l'âge des assurés", command=self.plot_histogram).pack(pady=10)
        ttk.Button(self.general_visual_window, text="Nombre de réclamations par état", command=self.plot_bar_chart).pack(pady=10)
        ttk.Button(self.general_visual_window, text="Distribution des réclamations totales", command=self.plot_total_claims_histogram).pack(pady=10)
        ttk.Button(self.general_visual_window, text="Heatmap des corrélations", command=self.plot_heatmap).pack(pady=10)
        ttk.Button(self.general_visual_window, text="Pie Chart - Fraud Reported", command=self.plot_fraud_reported_pie_chart).pack(pady=10)


    def open_fraud_visualization_window(self):
        self.fraud_visual_window = tk.Toplevel(self)
        self.fraud_visual_window.title("Fraud-related Data Visualizations")
        self.fraud_visual_window.geometry("300x200")
        self.fraud_visual_window.configure(background='#f0f0f0')

        ttk.Button(self.fraud_visual_window, text="Bar Chart - Fraud by State", command=self.plot_fraud_by_state).pack(pady=10)
        ttk.Button(self.fraud_visual_window, text="Pie Chart - Fraud Proportion", command=self.plot_fraud_proportion).pack(pady=10)
        ttk.Button(self.fraud_visual_window, text="Bar Chart - Fraud by Age Group", command=self.plot_fraud_by_age_group).pack(pady=10)
        ttk.Button(self.fraud_visual_window, text="Box Plot - Fraud by Capital Gains", command=self.plot_fraud_by_capital_gain).pack(pady=10)
        ttk.Button(self.fraud_visual_window, text="Box Plot - Fraud by Months as Customer", command=self.plot_fraud_by_months_as_customer).pack(pady=10)
        ttk.Button(self.fraud_visual_window, text="Box Plot - Fraud by Umbrella Limit", command=self.plot_fraud_by_umbrella_limit).pack(pady=10)
        ttk.Button(self.fraud_visual_window, text="Box Plot - Fraud by Capital Loss", command=self.plot_fraud_by_capital_loss).pack(pady=10)
        ttk.Button(self.fraud_visual_window, text="Count Plot - Fraud by Incident Hour", command=self.plot_fraud_by_incident_hour).pack(pady=10)
        ttk.Button(self.fraud_visual_window, text="Count Plot - Fraud by Bodily Injuries", command=self.plot_fraud_by_bodily_injuries).pack(pady=10)
        ttk.Button(self.fraud_visual_window, text="Box Plot - Fraud by Vehicle Claim", command=self.plot_fraud_by_vehicle_claim).pack(pady=10)
        ttk.Button(self.general_visual_window, text="Pie Chart - Fraud Reported", command=self.plot_fraud_reported_pie_chart).pack(pady=10)




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


    def plot_heatmap(self):
        data = self.db_manager.fetch_all_data()
        columns_to_keep = [
            "months_as_customer", "age", "policy_deductable", "policy_annual_premium", "umbrella_limit",
            "capital_gains", "capital_loss", "incident_hour_of_the_day", "number_of_vehicles_involved",
            "bodily_injuries", "witnesses", "total_claim_amount", "injury_claim", "property_claim", "vehicle_claim"
        ]
        data = data[columns_to_keep]
        
        # Handling non-numeric data by encoding
        for column in data.select_dtypes(include=['object']).columns:
            try:
                data[column] = data[column].astype(float)
            except ValueError:
                data[column] = LabelEncoder().fit_transform(data[column])

    def plot_fraud_by_state(self):
        data = self.db_manager.fetch_fraud_by_state()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.barplot(x=data['state'], y=data['fraud_count'], ax=ax)
        ax.set_title('Nombre de fraudes par état')
        ax.set_xlabel('État')
        ax.set_ylabel('Nombre de fraudes')
        self.display_figure(fig)

    def plot_fraud_proportion(self):
        data = self.db_manager.fetch_fraud_proportion()
        fig, ax = plt.subplots(figsize=(8, 6))
        labels = 'Fraud', 'No Fraud'
        sizes = [data['fraud'], data['no_fraud']]
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['red', 'blue'])
        ax.set_title('Proportion de fraudes')
        self.display_figure(fig)

    def plot_fraud_reported_pie_chart(self):
        data = self.db_manager.fetch_all_data()
        fraud_counts = data['fraud_reported'].value_counts()
        
        # Create the pie chart
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.pie(fraud_counts, labels=fraud_counts.index, autopct='%1.1f%%', startangle=90, colors=['blue', 'red'])
        ax.set_title('Répartition des fraudes signalées')
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        self.display_figure(fig)


    def plot_fraud_by_age_group(self):
        data = self.db_manager.fetch_all_data()
        # Encode 'fraud_reported' as 0 (No) and 1 (Yes)
        data['fraud_reported'] = data['fraud_reported'].map({'N': 0, 'Y': 1})
        
        # Create age groups
        bins = list(range(18, 70, 5))  # Creating bins from 0 to 100 with steps of 5
        labels = [f'{i}-{i+4}' for i in bins[:-1]]
        data['age_group'] = pd.cut(data['age'], bins=bins, labels=labels, right=False)
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(14, 8))
        sns.countplot(x='age_group', hue='fraud_reported', data=data, palette='coolwarm', ax=ax)
        ax.set_title('Nombre de fraudes par tranche d\'âge')
        ax.set_xlabel('Tranche d\'âge')
        ax.set_ylabel('Nombre de cas')
        ax.legend(title='Fraud Reported', labels=['No', 'Yes'])
        self.display_figure(fig)

    def plot_fraud_by_capital_gain(self):
        data = self.db_manager.fetch_all_data()
        # Encode 'fraud_reported' as 0 (No) and 1 (Yes)
        data['fraud_reported'] = data['fraud_reported'].map({'N': 0, 'Y': 1})
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(14, 8))
        sns.boxplot(x='fraud_reported', y='capital_gains', data=data, palette='coolwarm', ax=ax)
        ax.set_title('Répartition des gains en capital par rapport aux fraudes')
        ax.set_xlabel('Fraud Reported')
        ax.set_ylabel('Capital Gains')
        ax.set_xticklabels(['No', 'Yes'])
        self.display_figure(fig)

    def plot_fraud_by_months_as_customer(self):
        data = self.db_manager.fetch_all_data()
        # Encode 'fraud_reported' as 0 (No) and 1 (Yes)
        data['fraud_reported'] = data['fraud_reported'].map({'N': 0, 'Y': 1})
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(14, 8))
        sns.boxplot(x='fraud_reported', y='months_as_customer', data=data, palette='coolwarm', ax=ax)
        ax.set_title('Répartition des mois en tant que client par rapport aux fraudes')
        ax.set_xlabel('Fraud Reported')
        ax.set_ylabel('Months as Customer')
        ax.set_xticklabels(['No', 'Yes'])
        self.display_figure(fig)

    def plot_fraud_by_umbrella_limit(self):
        data = self.db_manager.fetch_all_data()
        # Encode 'fraud_reported' as 0 (No) and 1 (Yes)
        data['fraud_reported'] = data['fraud_reported'].map({'N': 0, 'Y': 1})
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(14, 8))
        sns.boxplot(x='fraud_reported', y='umbrella_limit', data=data, palette='coolwarm', ax=ax)
        ax.set_title('Répartition des limites d\'assurance par rapport aux fraudes')
        ax.set_xlabel('Fraud Reported')
        ax.set_ylabel('Umbrella Limit')
        ax.set_xticklabels(['No', 'Yes'])
        self.display_figure(fig)

    def plot_fraud_by_capital_loss(self):
        data = self.db_manager.fetch_all_data()
        # Encode 'fraud_reported' as 0 (No) and 1 (Yes)
        data['fraud_reported'] = data['fraud_reported'].map({'N': 0, 'Y': 1})
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(14, 8))
        sns.boxplot(x='fraud_reported', y='capital_loss', data=data, palette='coolwarm', ax=ax)
        ax.set_title('Répartition des pertes en capital par rapport aux fraudes')
        ax.set_xlabel('Fraud Reported')
        ax.set_ylabel('Capital Loss')
        ax.set_xticklabels(['No', 'Yes'])
        self.display_figure(fig)


    def display_figure(self, fig):
        visual_window = tk.Toplevel(self)
        visual_window.title("Visualization")
        visual_window.geometry("800x600")
        canvas = FigureCanvasTkAgg(fig, master=visual_window)
        canvas.draw()
        canvas.get_tk_widget().pack(expand=True)

    def plot_fraud_by_incident_hour(self):
        data = self.db_manager.fetch_all_data()
        # Encode 'fraud_reported' as 0 (No) and 1 (Yes)
        data['fraud_reported'] = data['fraud_reported'].map({'N': 0, 'Y': 1})
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(14, 8))
        sns.countplot(x='incident_hour_of_the_day', hue='fraud_reported', data=data, palette='coolwarm', ax=ax)
        ax.set_title('Répartition des fraudes par heure de l\'incident')
        ax.set_xlabel('Heure de l\'incident')
        ax.set_ylabel('Nombre de cas')
        ax.legend(title='Fraud Reported', labels=['No', 'Yes'])
        self.display_figure(fig)

    def plot_fraud_by_bodily_injuries(self):
        data = self.db_manager.fetch_all_data()
        # Encode 'fraud_reported' as 0 (No) and 1 (Yes)
        data['fraud_reported'] = data['fraud_reported'].map({'N': 0, 'Y': 1})
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(14, 8))
        sns.countplot(x='bodily_injuries', hue='fraud_reported', data=data, palette='coolwarm', ax=ax)
        ax.set_title('Répartition des fraudes par blessures corporelles')
        ax.set_xlabel('Blessures corporelles')
        ax.set_ylabel('Nombre de cas')
        ax.legend(title='Fraud Reported', labels=['No', 'Yes'])
        self.display_figure(fig)

    def plot_fraud_by_vehicle_claim(self):
        data = self.db_manager.fetch_all_data()
        # Encode 'fraud_reported' as 0 (No) and 1 (Yes)
        data['fraud_reported'] = data['fraud_reported'].map({'N': 0, 'Y': 1})
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(14, 8))
        sns.boxplot(x='fraud_reported', y='vehicle_claim', data=data, palette='coolwarm', ax=ax)
        ax.set_title('Répartition des réclamations de véhicules par rapport aux fraudes')
        ax.set_xlabel('Fraud Reported')
        ax.set_ylabel('Vehicle Claim')
        ax.set_xticklabels(['No', 'Yes'])
        self.display_figure(fig)



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
