import requests

def download_data(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Vérifie que la requête a réussi
        return response.json()  # Retourne les données JSON depuis la réponse
    except requests.RequestException as e:
        print(f"Erreur lors du téléchargement des données : {e}")
        return None
