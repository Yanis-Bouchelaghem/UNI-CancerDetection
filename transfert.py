import os
import shutil
from pathlib import Path

# Chemins de base
source_base = r"..\\dataset\\temp-test_dataset_patches"
target_base = r"..\\dataset\\test_dataset_patches"
def copy_and_rename_images():    
    # Créer les dossiers cibles s'ils n'existent paso
    target_cancereux = os.path.join(target_base, "tissu_cancereux")
    target_saint = os.path.join(target_base, "tissus_saint")
    
    os.makedirs(target_cancereux, exist_ok=True)
    os.makedirs(target_saint, exist_ok=True)
    
    # Parcourir tous les dossiers tumor_XXX
    if not os.path.exists(source_base):
        print(f"Erreur: Le dossier source {source_base} n'existe pas!")
        return
    
    tumor_folders = [f for f in os.listdir(source_base) if f.startswith("tumor_")]
    
    if not tumor_folders:
        print("Aucun dossier tumor_ trouvé dans le répertoire source!")
        return
    
    print(f"Trouvé {len(tumor_folders)} dossiers tumor à traiter...")
    
    total_copied = 0
    
    for tumor_folder in sorted(tumor_folders):
        tumor_path = os.path.join(source_base, tumor_folder)
        
        if not os.path.isdir(tumor_path):
            continue
            
        print(f"\nTraitement du dossier: {tumor_folder}")
        
        # Chercher les sous-dossiers (variations possibles dans les noms)
        possible_cancer_names = ["tissu_cancereux", "tissu_cancereux"]
        possible_saint_names = ["tissu_sain", "tissus_saint"]
        
        cancer_folder = None
        saint_folder = None
        
        # Trouver le bon nom de dossier pour les tissus cancéreux
        for name in possible_cancer_names:
            cancer_path = os.path.join(tumor_path, name)
            if os.path.exists(cancer_path):
                cancer_folder = cancer_path
                break
        
        # Trouver le bon nom de dossier pour les tissus saints
        for name in possible_saint_names:
            saint_path = os.path.join(tumor_path, name)
            if os.path.exists(saint_path):
                saint_folder = saint_path
                break
        
        # Copier les fichiers des tissus cancéreux
        if cancer_folder and os.path.exists(cancer_folder):
            files = [f for f in os.listdir(cancer_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))]
            print(f"  - Tissu cancéreux: {len(files)} fichiers trouvés")
            
            for file in files:
                source_file = os.path.join(cancer_folder, file)
                new_filename = f"{tumor_folder}_{file}"
                target_file = os.path.join(target_cancereux, new_filename)
                
                try:
                    shutil.copy2(source_file, target_file)
                    total_copied += 1
                except Exception as e:
                    print(f"    Erreur lors de la copie de {file}: {e}")
        else:
            print(f"  - Dossier tissu cancéreux non trouvé dans {tumor_folder}")
        
        # Copier les fichiers des tissus saints
        if saint_folder and os.path.exists(saint_folder):
            files = [f for f in os.listdir(saint_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))]
            print(f"  - Tissu saint: {len(files)} fichiers trouvés")
            
            for file in files:
                source_file = os.path.join(saint_folder, file)
                new_filename = f"{tumor_folder}_{file}"
                target_file = os.path.join(target_saint, new_filename)
                
                try:
                    shutil.copy2(source_file, target_file)
                    total_copied += 1
                except Exception as e:
                    print(f"    Erreur lors de la copie de {file}: {e}")
        else:
            print(f"  - Dossier tissu saint non trouvé dans {tumor_folder}")
    
    print(f"\n=== Copie terminée ===")
    print(f"Total de fichiers copiés: {total_copied}")
    print(f"Dossier de destination: {target_base}")

def verify_structure():
    """Fonction pour vérifier la structure des dossiers avant la copie"""
    
    print("=== Vérification de la structure ===")
    
    if not os.path.exists(source_base):
        print(f"Le dossier source {source_base} n'existe pas!")
        return
    
    tumor_folders = [f for f in os.listdir(source_base) if f.startswith("tumor_")]
    print(f"Dossiers tumor trouvés: {len(tumor_folders)}")
    
    for tumor_folder in sorted(tumor_folders[:3]):  # Afficher seulement les 3 premiers
        tumor_path = os.path.join(source_base, tumor_folder)
        print(f"\n{tumor_folder}:")
        
        if os.path.isdir(tumor_path):
            subdirs = [d for d in os.listdir(tumor_path) if os.path.isdir(os.path.join(tumor_path, d))]
            for subdir in subdirs:
                subdir_path = os.path.join(tumor_path, subdir)
                file_count = len([f for f in os.listdir(subdir_path) 
                                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))])
                print(f"  - {subdir}: {file_count} images")

if __name__ == "__main__":
    print("Script de copie et renommage d'images médicales")
    print("=" * 50)
    
    # Vérifier d'abord la structure
    verify_structure()
    
    print("\n" + "=" * 50)
    response = input("Voulez-vous procéder à la copie? (o/n): ")
    
    if response.lower() in ['o', 'oui', 'y', 'yes']:
        copy_and_rename_images()
    else:
        print("Opération annulée.")