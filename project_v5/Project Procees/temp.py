import os

# Définir le chemin du répertoire où créer les dossiers
path = os.getcwd()

# Itérer sur les numéros de 1 à 12
for i in range(1, 13):
    # Créer le nom du dossier avec le numéro actuel
    folder_name = f"INTG6_traces_{i}"

    # Vérifier si le dossier existe déjà
    if not os.path.exists(folder_name):
        # Créer le dossier s'il n'existe pas
        os.makedirs(folder_name)
        print(f"Dossier créé : {folder_name}")
    else:
        print(f"Le dossier '{folder_name}' existe déjà.")
