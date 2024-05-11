import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from database import DatabaseManager
import pandas as pd
import numpy as np
import seaborn as sns  # Importer Seaborn pour les graphiques avancés

class MainApplication(tk.Tk):
    def __init__(self):
        super().__init__()
        self.db_manager = DatabaseManager('database_insurance_claims.db')
        self.style = ttk.Style(self)
        self.style.theme_use('clam')  # Choisissez le thème qui convient le mieux à votre application
        self.title("Insurance Claims Application")
        self.geometry("800x600")
        self.initialize_ui()
        sns.set(style="whitegrid")  # Configurer le style Seaborn ici pour que tout le graphique l'utilise

    def initialize_ui(self):
        self.configure(background='#f0f0f0')
        header_frame = ttk.Frame(self, padding="10 10 10 5")
        header_frame.pack(fill=tk.X, expand=False)

        ttk.Label(header_frame, text="Insurance Claims Visualization", font=("Helvetica", 18, "bold"), background='#f0f0f0').pack(side=tk.LEFT, padx=10)

        content_frame = ttk.Frame(self, padding="3 3 12 12")
        content_frame.pack(fill=tk.BOTH, expand=True)

        # Bouton pour ouvrir la fenêtre de visualisation
        visual_button = ttk.Button(content_frame, text="Visualisation", command=self.open_visualization_window)
        visual_button.pack(pady=20)

    def open_visualization_window(self):
        self.visual_window = tk.Toplevel(self)
        self.visual_window.title("Data Visualizations")
        self.visual_window.geometry("800x600")
        self.visual_window.configure(background='#f0f0f0')

        # Boutons pour différents types de graphiques
        ttk.Button(self.visual_window, text="Plot Data", command=self.plot_data).pack(pady=10)
        ttk.Button(self.visual_window, text="Histogram", command=self.plot_histogram).pack(pady=10)
        ttk.Button(self.visual_window, text="Scatter Plot", command=self.plot_scatter).pack(pady=10)
        ttk.Button(self.visual_window, text="Bar Chart - Claims by State", command=self.plot_bar_chart).pack(pady=10)
        ttk.Button(self.visual_window, text="Histogram - Total Claims", command=self.plot_total_claims_histogram).pack(pady=10)


    def plot_data(self):
        data = self.db_manager.fetch_some_data()  # Placeholder function
        fig, ax = plt.subplots()
        ax.plot(data['x'], data['y'])
        self.display_figure(fig)

    def plot_histogram(self):
        data = self.db_manager.fetch_histogram_data()  # Cette fonction doit être définie pour récupérer les données nécessaires
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.histplot(data['age'], bins=30, color='blue', kde=True, ax=ax)
        ax.set_title('Distribution de l\'âge des assurés')
        ax.set_xlabel('Âge')
        ax.set_ylabel('Fréquence')
        self.display_figure(fig)

    def plot_bar_chart(self):
        data = self.db_manager.fetch_claims_by_state()  # Cette fonction doit être définie pour récupérer les données nécessaires
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.barplot(x=data['state'], y=data['claims_count'], ax=ax)
        ax.set_title('Nombre de réclamations par état')
        ax.set_xlabel('État')
        ax.set_ylabel('Nombre de réclamations')
        self.display_figure(fig)

    def plot_total_claims_histogram(self):
        data = self.db_manager.fetch_total_claims_data()  # Assurez-vous que cette fonction récupère les montants des réclamations totales
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

