import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from ctran import ctranspath
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
trnsfrms_val = transforms.Compose([
    # transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

class ImageDataset(Dataset):
    """Dataset pour charger les images avec leurs labels"""
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]

def load_image_paths_and_labels(data_dir):
    """
    Charge les chemins d'images et leurs labels depuis les dossiers
    
    Args:
        data_dir (str): Chemin vers le dossier contenant tissus_cancereux et tissus_sain
    
    Returns:
        tuple: (image_paths, labels)
    """
    image_paths = []
    labels = []
    
    # Dossier tissus cancéreux (label = 1)
    cancer_dir = os.path.join(data_dir, "tissu_cancereux")
    if os.path.exists(cancer_dir):
        for filename in os.listdir(cancer_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
                image_paths.append(os.path.join(cancer_dir, filename))
                labels.append(1)  # Cancéreux
    
    # Dossier tissus sains (label = 0)
    healthy_dir = os.path.join(data_dir, "tissu_sain")
    if os.path.exists(healthy_dir):
        for filename in os.listdir(healthy_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
                image_paths.append(os.path.join(healthy_dir, filename))
                labels.append(0)  # Sain
    
    print(f"Images chargées: {len(image_paths)} total")
    print(f"- Tissus cancéreux: {labels.count(1)}")
    print(f"- Tissus sains: {labels.count(0)}")
    
    return image_paths, labels

def plot_data_distribution(labels, save_dir="./results"):
    """
    Affiche la distribution des classes
    """
    plt.figure(figsize=(8, 6))
    unique, counts = np.unique(labels, return_counts=True)
    class_names = ['Tissus Sains', 'Tissus Cancéreux']
    colors = ['lightblue', 'lightcoral']
    
    bars = plt.bar(class_names, counts, color=colors, alpha=0.8, edgecolor='black')
    plt.title('Distribution des Classes dans le Dataset', fontsize=16, fontweight='bold')
    plt.ylabel('Nombre d\'échantillons', fontsize=12)
    plt.xlabel('Classes', fontsize=12)
    
    # Ajouter les valeurs sur les barres
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                str(count), ha='center', va='bottom', fontweight='bold')
    
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'data_distribution.png'), dpi=300, bbox_inches='tight')
    plt.show()

def load_ctranspath_model(model_path, device=None):
    """
    Charge le modèle CTransPath pour extraction de features
    
    Args:
        model_path (str): Chemin vers ctranspath.pth
        device: Device à utiliser (None pour auto-détection)
    
    Returns:
        torch.nn.Module: Modèle CTransPath configuré pour extraction
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = ctranspath()
    model.head = nn.Identity()  # Supprimer la couche de classification
    
    # Charger sur le bon device dès le début
    td = torch.load(model_path, map_location=device)
    model.load_state_dict(td['model'], strict=True)
    model = model.to(device)
    model.eval()
    
    print(f"Modèle CTransPath chargé avec succès sur {device}")
    return model

def extract_embeddings(model, image_paths, labels, batch_size=16, use_gpu=True, save_cache=True, cache_file="embeddings_cache.npz"):
    """
    Extrait les embeddings de toutes les images avec optimisations
    
    Args:
        model: Modèle CTransPath
        image_paths: Liste des chemins d'images
        labels: Liste des labels
        batch_size: Taille du batch (augmentée pour plus d'efficacité)
        use_gpu: Utiliser le GPU si disponible
        save_cache: Sauvegarder les embeddings pour éviter de les recalculer
        cache_file: Fichier de cache
    
    Returns:
        tuple: (embeddings_array, labels_array)
    """
    # Vérifier si un cache existe
    if save_cache and os.path.exists(cache_file):
        print(f"📁 Cache trouvé : {cache_file}")
        response = input("Voulez-vous charger les embeddings depuis le cache ? (y/n): ")
        if response.lower() == 'y':
            data = np.load(cache_file)
            print("✅ Embeddings chargés depuis le cache")
            return data['embeddings'], data['labels']
    
    # Configuration GPU - s'assurer que le device est cohérent
    device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
    print(f"🔧 Utilisation de : {device}")
    
    # S'assurer que le modèle est sur le bon device
    model = model.to(device)
    
    dataset = ImageDataset(image_paths, labels, transform=trnsfrms_val)
    # Utiliser plus de workers pour le chargement parallèle
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                          num_workers=4, pin_memory=True if device.type == 'cuda' else False)
    
    all_embeddings = []
    all_labels = []
    
    print("🔄 Extraction des embeddings...")
    import time
    start_time = time.time()
    
    with torch.no_grad():
        for i, (images, batch_labels) in enumerate(dataloader):
            # CRUCIAL: S'assurer que les images sont sur le même device que le modèle
            images = images.to(device, non_blocking=True)
            
            # Le modèle et les images sont maintenant sur le même device
            embeddings = model(images)
            
            # Ramener sur CPU pour stockage
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.extend(batch_labels.numpy())
            
            # Affichage du progrès avec estimation du temps
            if (i + 1) % 5 == 0:
                elapsed = time.time() - start_time
                estimated_total = elapsed * len(dataloader) / (i + 1)
                remaining = estimated_total - elapsed
                print(f"📊 Batch {i + 1}/{len(dataloader)} - Temps restant: ~{remaining/60:.1f}min")
    
    # Concaténer tous les embeddings
    embeddings_array = np.vstack(all_embeddings)
    labels_array = np.array(all_labels)
    
    # Sauvegarder dans le cache
    if save_cache:
        np.savez_compressed(cache_file, embeddings=embeddings_array, labels=labels_array)
        print(f"💾 Embeddings sauvegardés dans le cache : {cache_file}")
    
    total_time = time.time() - start_time
    print(f"✅ Embeddings extraits: {embeddings_array.shape} en {total_time/60:.1f}min")
    return embeddings_array, labels_array

class SimpleMLP(nn.Module):
    """MLP simple pour classification binaire"""
    def __init__(self, input_dim=768, hidden_dim=256, num_classes=2, dropout=0.3):
        super(SimpleMLP, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, x):
        return self.classifier(x)

def create_mlp_model(input_dim=768, hidden_dim=256, num_classes=2):
    """
    Crée un modèle MLP
    
    Args:
        input_dim: Dimension des embeddings (768 pour CTransPath)
        hidden_dim: Dimension de la couche cachée
        num_classes: Nombre de classes (2 pour binaire)
    
    Returns:
        torch.nn.Module: Modèle MLP
    """
    model = SimpleMLP(input_dim, hidden_dim, num_classes)
    print(f"MLP créé: {input_dim} -> {hidden_dim} -> {hidden_dim//2} -> {num_classes}")
    return model

def prepare_data_for_training(embeddings, labels, test_size=0.2, random_state=42):
    """
    Prépare les données pour l'entraînement
    
    Args:
        embeddings: Array des embeddings
        labels: Array des labels
        test_size: Proportion pour le test
        random_state: Seed pour la reproductibilité
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    
    print(f"Données d'entraînement: {X_train.shape[0]} échantillons")
    print(f"Données de test: {X_test.shape[0]} échantillons")
    
    return X_train, X_test, y_train, y_test

def train_mlp(model, X_train, y_train, X_test, y_test, epochs=50, lr=0.001, batch_size=64, early_stopping=True, patience=10):
    """
    Entraîne le MLP avec optimisations
    
    Args:
        model: Modèle MLP
        X_train, y_train: Données d'entraînement
        X_test, y_test: Données de test
        epochs: Nombre d'époques maximum
        lr: Taux d'apprentissage
        batch_size: Taille du batch (augmentée)
        early_stopping: Arrêt anticipé si pas d'amélioration
        patience: Nombre d'époques sans amélioration avant arrêt
    
    Returns:
        dict: Historique de l'entraînement
    """
    # Configuration GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Entraînement sur : {device}")
    model = model.to(device)
    
    # Conversion en tenseurs
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.LongTensor(y_train).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_test_tensor = torch.LongTensor(y_test).to(device)
    
    # DataLoader pour l'entraînement
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Optimiseur et fonction de perte
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)  # Ajout de régularisation
    
    history = {'train_loss': [], 'train_acc': [], 'test_acc': []}
    best_test_acc = 0
    patience_counter = 0
    
    print("🚀 Début de l'entraînement...")
    import time
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        epoch_correct = 0
        epoch_total = 0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            epoch_total += batch_y.size(0)
            epoch_correct += (predicted == batch_y).sum().item()
        
        # Évaluation sur le test
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_tensor)
            _, test_predicted = torch.max(test_outputs.data, 1)
            test_acc = (test_predicted == y_test_tensor).sum().item() / len(y_test_tensor)
        
        train_acc = epoch_correct / epoch_total
        avg_loss = epoch_loss / len(train_loader)
        
        history['train_loss'].append(avg_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        
        # Early stopping
        if early_stopping:
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                patience_counter = 0
                # Sauvegarder le meilleur modèle
                torch.save(model.state_dict(), 'best_model_temp.pth')
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"🛑 Arrêt anticipé à l'époque {epoch+1}")
                # Charger le meilleur modèle
                model.load_state_dict(torch.load('best_model_temp.pth'))
                os.remove('best_model_temp.pth')  # Nettoyer
                break
        
        if (epoch + 1) % 5 == 0:  # Affichage plus fréquent
            elapsed = time.time() - start_time
            estimated_total = elapsed * epochs / (epoch + 1)
            remaining = estimated_total - elapsed
            print(f"📊 Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Train: {train_acc:.4f} - Test: {test_acc:.4f} - Restant: {remaining/60:.1f}min")
    
    total_time = time.time() - start_time
    print(f"✅ Entraînement terminé en {total_time/60:.1f}min")
    
    return history

def plot_training_history(history, save_dir="./results"):
    """
    Affiche les courbes d'entraînement
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Courbe de perte
    epochs = range(1, len(history['train_loss']) + 1)
    ax1.plot(epochs, history['train_loss'], 'b-', linewidth=2, label='Perte d\'entraînement')
    ax1.set_title('Évolution de la Perte', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Époques')
    ax1.set_ylabel('Perte')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Courbes de précision
    ax2.plot(epochs, history['train_acc'], 'b-', linewidth=2, label='Précision d\'entraînement')
    ax2.plot(epochs, history['test_acc'], 'r-', linewidth=2, label='Précision de test')
    ax2.set_title('Évolution de la Précision', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Époques')
    ax2.set_ylabel('Précision')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
    plt.show()

def plot_confusion_matrix(y_true, y_pred, save_dir="./results"):
    """
    Affiche la matrice de confusion
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Sain', 'Cancéreux'],
                yticklabels=['Sain', 'Cancéreux'],
                cbar_kws={'label': 'Nombre de prédictions'})
    
    plt.title('Matrice de Confusion', fontsize=16, fontweight='bold')
    plt.xlabel('Prédictions', fontsize=12)
    plt.ylabel('Valeurs Réelles', fontsize=12)
    
    # Ajouter des statistiques
    accuracy = (cm[0,0] + cm[1,1]) / np.sum(cm)
    precision_healthy = cm[0,0] / (cm[0,0] + cm[1,0]) if (cm[0,0] + cm[1,0]) > 0 else 0
    precision_cancer = cm[1,1] / (cm[1,1] + cm[0,1]) if (cm[1,1] + cm[0,1]) > 0 else 0
    recall_healthy = cm[0,0] / (cm[0,0] + cm[0,1]) if (cm[0,0] + cm[0,1]) > 0 else 0
    recall_cancer = cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0
    
    plt.figtext(0.02, 0.02, f'Précision globale: {accuracy:.3f}', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.show()

def plot_metrics_summary(y_true, y_pred, save_dir="./results"):
    """
    Affiche un résumé des métriques par classe
    """
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    # Calcul des métriques pour chaque classe
    precision = precision_score(y_true, y_pred, average=None)
    recall = recall_score(y_true, y_pred, average=None)
    f1 = f1_score(y_true, y_pred, average=None)
    
    classes = ['Tissus Sains', 'Tissus Cancéreux']
    metrics = ['Précision', 'Rappel', 'F1-Score']
    
    # Création du graphique
    x = np.arange(len(classes))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars1 = ax.bar(x - width, precision, width, label='Précision', alpha=0.8, color='skyblue')
    bars2 = ax.bar(x, recall, width, label='Rappel', alpha=0.8, color='lightgreen')
    bars3 = ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8, color='lightcoral')
    
    ax.set_xlabel('Classes', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Métriques de Performance par Classe', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Ajouter les valeurs sur les barres
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    add_value_labels(bars1)
    add_value_labels(bars2)
    add_value_labels(bars3)
    
    plt.ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'metrics_summary.png'), dpi=300, bbox_inches='tight')
    plt.show()

def evaluate_model(model, X_test, y_test, save_dir="./results"):
    """
    Évalue le modèle final avec visualisations
    
    Args:
        model: Modèle entraîné
        X_test, y_test: Données de test
        save_dir: Dossier pour sauvegarder les graphiques
    
    Returns:
        dict: Métriques d'évaluation
    """
    # S'assurer que le modèle est en mode évaluation
    model.eval()
    
    # Déterminer le device du modèle
    device = next(model.parameters()).device
    print(f"🔧 Évaluation sur device: {device}")
    
    # S'assurer que les données de test sont sur le même device que le modèle
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    
    with torch.no_grad():
        outputs = model(X_test_tensor)
        _, predicted = torch.max(outputs.data, 1)
        # Ramener les prédictions sur CPU pour les calculs de métriques
        predicted = predicted.cpu().numpy()
    
    accuracy = accuracy_score(y_test, predicted)
    report = classification_report(y_test, predicted, target_names=['Sain', 'Cancéreux'])
    
    print(f"\n=== RÉSULTATS FINAUX ===")
    print(f"Précision: {accuracy:.4f}")
    print(f"\nRapport détaillé:")
    print(report)
    
    # Générer les visualisations
    os.makedirs(save_dir, exist_ok=True)
    plot_confusion_matrix(y_test, predicted, save_dir)
    plot_metrics_summary(y_test, predicted, save_dir)
    
    return {'accuracy': accuracy, 'predictions': predicted, 'report': report}

def save_model_and_results(model, history, results, save_dir="./results"):
    """
    Sauvegarde le modèle et les résultats
    
    Args:
        model: Modèle entraîné
        history: Historique d'entraînement
        results: Résultats d'évaluation
        save_dir: Dossier de sauvegarde
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Sauvegarder le modèle
    torch.save(model.state_dict(), os.path.join(save_dir, "mlp_model.pth"))
    
    # Sauvegarder l'historique
    np.save(os.path.join(save_dir, "training_history.npy"), history)
    
    # Sauvegarder les résultats au format texte
    with open(os.path.join(save_dir, "results_summary.txt"), "w") as f:
        f.write("=== RÉSULTATS DE L'ENTRAÎNEMENT ===\n\n")
        f.write(f"Précision finale: {results['accuracy']:.4f}\n\n")
        f.write("Rapport de classification:\n")
        f.write(results['report'])
    
    print(f"Modèle et résultats sauvegardés dans {save_dir}")

# ======================= PIPELINE PRINCIPAL =======================

def main():
    """Pipeline principal d'entraînement"""
    
    # Configuration
    DATA_DIR = "d:/wsi/extracted_regions_batch"  # Ajustez selon votre structure
    CTRANSPATH_MODEL = "./ctranspath.pth"
    SAVE_DIR = "./results224"
    
    # Créer le dossier de résultats
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # 1. Charger les chemins d'images et labels
    print("=== ÉTAPE 1: Chargement des données ===")
    image_paths, labels = load_image_paths_and_labels(DATA_DIR)
    
    # Visualiser la distribution des données
    plot_data_distribution(labels, SAVE_DIR)
    
    # 2. Charger le modèle CTransPath
    print("\n=== ÉTAPE 2: Chargement du modèle CTransPath ===")
    ctrans_model = load_ctranspath_model(CTRANSPATH_MODEL)
    
    # 3. Extraire les embeddings
    print("\n=== ÉTAPE 3: Extraction des embeddings ===")
    embeddings, labels_array = extract_embeddings(ctrans_model, image_paths, labels)
    
    # 4. Préparer les données pour l'entraînement
    print("\n=== ÉTAPE 4: Préparation des données ===")
    X_train, X_test, y_train, y_test = prepare_data_for_training(embeddings, labels_array)
    
    # 5. Créer le modèle MLP
    print("\n=== ÉTAPE 5: Création du MLP ===")
    mlp_model = create_mlp_model(input_dim=embeddings.shape[1])
    
    # 6. Entraîner le modèle
    print("\n=== ÉTAPE 6: Entraînement ===")
    history = train_mlp(mlp_model, X_train, y_train, X_test, y_test)
    
    # Visualiser l'historique d'entraînement
    plot_training_history(history, SAVE_DIR)
    
    # 7. Évaluer le modèle
    print("\n=== ÉTAPE 7: Évaluation ===")
    results = evaluate_model(mlp_model, X_test, y_test, SAVE_DIR)
    
    # 8. Sauvegarder
    print("\n=== ÉTAPE 8: Sauvegarde ===")
    save_model_and_results(mlp_model, history, results, SAVE_DIR)
    
    print("\n🎉 Entraînement terminé avec succès!")
    print(f"📊 Tous les graphiques ont été sauvegardés dans le dossier: {SAVE_DIR}")

if __name__ == "__main__":
    main()