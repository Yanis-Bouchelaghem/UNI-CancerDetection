import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import pickle
import os
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
from ctran import ctranspath
from torchvision import transforms
import pandas as pd
from tqdm import tqdm
import time
from pathlib import Path
import cv2

# Importer votre extracteur de patches
from WSICancerPatchExtractor import WSICancerPatchExtractor, process_single_wsi

class SimpleMLP(torch.nn.Module):
    """MLP simple pour classification binaire - même architecture que l'entraînement"""
    def __init__(self, input_dim=768, hidden_dim=256, num_classes=2, dropout=0.3):
        super(SimpleMLP, self).__init__()
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim // 2, num_classes)
        )
   
    def forward(self, x):
        return self.classifier(x)

class WSITestPipeline:
    def __init__(self, mlp_model_path, ctranspath_model_path):
        """
        Pipeline de test utilisant les patches extraits de WSI
        
        Args:
            mlp_model_path: Chemin vers le modèle MLP (.pth)
            ctranspath_model_path: Chemin vers le modèle CTransPath (.pth)
        """
        self.mlp_model_path = mlp_model_path
        self.ctranspath_model_path = ctranspath_model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔧 Utilisation de : {self.device}")
        
        # Charger le modèle MLP
        print("📚 Chargement du modèle MLP...")
        self.mlp_model = SimpleMLP(input_dim=768, hidden_dim=256, num_classes=2, dropout=0.3)
        state_dict = torch.load(mlp_model_path, map_location=self.device)
        self.mlp_model.load_state_dict(state_dict)
        self.mlp_model = self.mlp_model.to(self.device).eval()
        print("✅ Modèle MLP chargé")
        
        # Charger CTransPath
        print("📚 Chargement du modèle CTransPath...")
        self.ctrans_net = ctranspath()
        td = torch.load(ctranspath_model_path, map_location=self.device)
        self.ctrans_net.load_state_dict(td['model'], strict=False)
        self.ctrans_net = self.ctrans_net.to(self.device).eval()
        self.ctrans_net.head = torch.nn.Identity()
        print("✅ Modèle CTransPath chargé")
        
        # Transform identique à l'entraînement
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def extract_patches_from_wsi(self, wsi_path, xml_path, patch_size=256, padding=200, 
                                temp_dir="temp_patches"):
        """
        Extraire patches depuis WSI en utilisant vos fonctions
        
        Args:
            wsi_path: Chemin vers le fichier WSI (.tif)
            xml_path: Chemin vers le fichier XML d'annotations
            patch_size: Taille des patches
            padding: Padding autour de la bbox
            temp_dir: Dossier temporaire pour les patches
        
        Returns:
            patches_data: Liste de dictionnaires avec patch et métadonnées
        """
        print(f"🔄 Extraction des patches depuis WSI...")
        print(f"   WSI: {wsi_path}")
        print(f"   XML: {xml_path}")
        
        # Créer l'extracteur
        extractor = WSICancerPatchExtractor(wsi_path, xml_path, patch_size)
        
        # Charger la WSI
        extractor.load_wsi()
        
        # Parser les annotations
        annotations = extractor.parse_xml_annotations()
        
        if results:
            print("\n🎉 TEST WSI TERMINÉ AVEC SUCCÈS!")
            print(f"📁 Résultats complets sauvegardés dans: {SAVE_DIR}")
        
        # Résumé final pour le médecin
        metrics = results['metrics']
        print("\n" + "="*60)
        print("📋 RÉSUMÉ MÉDICAL")
        print("="*60)
        print(f"🎯 Précision globale: {metrics['accuracy']*100:.1f}%")
        print(f"🔍 Sensibilité cancer: {metrics['recall_per_class'][1]*100:.1f}%")
        print(f"🎯 Précision cancer: {metrics['precision_per_class'][1]*100:.1f}%")
        print(f"⚖️ F1-Score cancer: {metrics['f1_per_class'][1]*100:.1f}%")
        
        # Évaluation critique
        recall_cancer = metrics['recall_per_class'][1]
        if recall_cancer >= 0.95:
            print("\n✅ CONCLUSION: Modèle fiable pour l'aide au diagnostic")
        elif recall_cancer >= 0.90:
            print("\n⚠️ CONCLUSION: Performance acceptable mais nécessite surveillance")
        else:
            print("\n❌ CONCLUSION: Performance insuffisante pour usage clinique")
            
        tn, fp, fn, tp = metrics['confusion_matrix'].ravel()
        if fn > 0:
            print(f"⚠️ ATTENTION: {fn} cancer(s) non détecté(s) sur {tp + fn} total")
        
        return results, pipeline
    else:
        print("❌ Échec du test WSI!")
        return None, pipeline

def run_batch_wsi_test():
    """
    Fonction pour tester le modèle sur plusieurs WSI
    """
    # Liste des fichiers WSI à tester
    wsi_test_files = [
        (r'd:/wsi/tumor_001.tif', r'd:/wsi/annot/tumor_001.xml'),
        (r'd:/wsi/tumor_002.tif', r'd:/wsi/annot/tumor_002.xml'),
        # Ajoutez d'autres fichiers ici
    ]
    
    # Chemins vers vos modèles
    MLP_MODEL_PATH = r'D:\Python code\results\mlp_model.pth'
    CTRANSPATH_MODEL_PATH = r'./ctranspath.pth'
    
    # Paramètres
    PATCH_SIZE = 256
    PADDING = 200
    BASE_SAVE_DIR = './batch_wsi_test_results'
    
    print("🏥 DÉBUT DU TEST BATCH WSI")
    print("="*60)
    
    # Initialiser le pipeline une seule fois
    pipeline = WSITestPipeline(MLP_MODEL_PATH, CTRANSPATH_MODEL_PATH)
    
    # Résultats globaux
    all_results = []
    global_stats = {
        'total_files': len(wsi_test_files),
        'successful_files': 0,
        'failed_files': 0,
        'total_patches': 0,
        'total_cancer_patches': 0,
        'total_sain_patches': 0,
        'global_metrics': None
    }
    
    # Variables pour métriques globales
    all_true_labels = []
    all_predictions = []
    all_probabilities = []
    
    # Traiter chaque fichier WSI
    for i, (wsi_path, xml_path) in enumerate(wsi_test_files, 1):
        print(f"\n📄 FICHIER {i}/{len(wsi_test_files)}")
        print("-" * 40)
        
        # Créer un dossier spécifique pour ce fichier
        file_name = Path(wsi_path).stem
        save_dir = f"{BASE_SAVE_DIR}/{file_name}"
        
        try:
            # Tester ce fichier WSI
            results = pipeline.test_wsi_complete(
                wsi_path=wsi_path,
                xml_path=xml_path,
                save_dir=save_dir,
                patch_size=PATCH_SIZE,
                padding=PADDING
            )
            
            if results:
                all_results.append({
                    'file': file_name,
                    'wsi_path': wsi_path,
                    'xml_path': xml_path,
                    'status': 'success',
                    'results': results
                })
                
                # Accumuler pour statistiques globales
                global_stats['successful_files'] += 1
                global_stats['total_patches'] += results['processing_info']['processed_patches']
                
                # Accumuler les labels et prédictions pour métriques globales
                for detail in results['detailed_results']:
                    all_true_labels.append(detail['true_label'])
                    all_predictions.append(detail['predicted_label'])
                    all_probabilities.append([detail['prob_sain'], detail['prob_cancer']])
                
                print(f"✅ {file_name} traité avec succès")
                
            else:
                all_results.append({
                    'file': file_name,
                    'wsi_path': wsi_path,
                    'xml_path': xml_path,
                    'status': 'failed',
                    'error': 'Aucun résultat retourné'
                })
                global_stats['failed_files'] += 1
                print(f"❌ Échec pour {file_name}")
                
        except Exception as e:
            all_results.append({
                'file': file_name,
                'wsi_path': wsi_path,
                'xml_path': xml_path,
                'status': 'failed',
                'error': str(e)
            })
            global_stats['failed_files'] += 1
            print(f"❌ Erreur pour {file_name}: {e}")
    
    # Calculer les métriques globales si on a des données
    if all_true_labels:
        global_metrics = pipeline.calculate_wsi_metrics(
            all_true_labels, all_predictions, all_probabilities
        )
        global_stats['global_metrics'] = global_metrics
        
        # Statistiques par classe
        global_stats['total_cancer_patches'] = sum(all_true_labels)
        global_stats['total_sain_patches'] = len(all_true_labels) - sum(all_true_labels)
    
    # Sauvegarder le rapport global
    save_batch_report(all_results, global_stats, BASE_SAVE_DIR)
    
    # Afficher le résumé final
    print("\n" + "="*60)
    print("📊 RÉSUMÉ GLOBAL DU TEST BATCH")
    print("="*60)
    print(f"📁 Fichiers testés: {global_stats['total_files']}")
    print(f"✅ Succès: {global_stats['successful_files']}")
    print(f"❌ Échecs: {global_stats['failed_files']}")
    print(f"🔬 Total patches: {global_stats['total_patches']}")
    
    if global_stats['global_metrics']:
        gm = global_stats['global_metrics']
        print(f"\n🎯 PERFORMANCE GLOBALE:")
        print(f"   Accuracy: {gm['accuracy']*100:.1f}%")
        print(f"   F1-Score: {gm['f1_score']*100:.1f}%")
        print(f"   Recall Cancer: {gm['recall_per_class'][1]*100:.1f}%")
        print(f"   Precision Cancer: {gm['precision_per_class'][1]*100:.1f}%")
        
        # Analyse critique globale
        tn, fp, fn, tp = gm['confusion_matrix'].ravel()
        total_cancer = tp + fn
        if total_cancer > 0:
            miss_rate = fn / total_cancer * 100
            print(f"\n⚠️ ANALYSE CRITIQUE:")
            print(f"   Cancers ratés: {fn}/{total_cancer} ({miss_rate:.1f}%)")
            
            if miss_rate <= 5:
                print("   ✅ Taux d'erreur acceptable")
            elif miss_rate <= 10:
                print("   ⚠️ Taux d'erreur à surveiller")
            else:
                print("   ❌ Taux d'erreur trop élevé")
    
    print(f"\n📁 Rapports sauvegardés dans: {BASE_SAVE_DIR}")
    return all_results, global_stats, pipeline

def save_batch_report(all_results, global_stats, save_dir):
    """
    Sauvegarder le rapport complet du test batch
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Rapport JSON
    import json
    report_data = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'global_stats': {
            'total_files': global_stats['total_files'],
            'successful_files': global_stats['successful_files'],
            'failed_files': global_stats['failed_files'],
            'total_patches': global_stats['total_patches'],
            'total_cancer_patches': global_stats['total_cancer_patches'],
            'total_sain_patches': global_stats['total_sain_patches']
        },
        'file_results': []
    }
    
    # Ajouter les résultats par fichier
    for result in all_results:
        file_data = {
            'file': result['file'],
            'status': result['status']
        }
        
        if result['status'] == 'success':
            metrics = result['results']['metrics']
            file_data.update({
                'accuracy': float(metrics['accuracy']),
                'f1_score': float(metrics['f1_score']),
                'recall_cancer': float(metrics['recall_per_class'][1]),
                'precision_cancer': float(metrics['precision_per_class'][1]),
                'patches_processed': result['results']['processing_info']['processed_patches']
            })
        else:
            file_data['error'] = result.get('error', 'Unknown error')
        
        report_data['file_results'].append(file_data)
    
    # Ajouter métriques globales
    if global_stats['global_metrics']:
        gm = global_stats['global_metrics']
        report_data['global_metrics'] = {
            'accuracy': float(gm['accuracy']),
            'f1_score': float(gm['f1_score']),
            'precision': float(gm['precision']),
            'recall': float(gm['recall']),
            'recall_cancer': float(gm['recall_per_class'][1]),
            'precision_cancer': float(gm['precision_per_class'][1]),
            'confusion_matrix': gm['confusion_matrix'].tolist()
        }
    
    # Sauvegarder JSON
    with open(f"{save_dir}/batch_test_report.json", 'w') as f:
        json.dump(report_data, f, indent=2)
    
    # Rapport texte médical
    with open(f"{save_dir}/batch_medical_report.txt", 'w', encoding='utf-8') as f:
        f.write("=== RAPPORT MÉDICAL - TEST BATCH WSI ===\n\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Fichiers analysés: {global_stats['total_files']}\n")
        f.write(f"Succès: {global_stats['successful_files']}\n")
        f.write(f"Échecs: {global_stats['failed_files']}\n\n")
        
        f.write("=== RÉSULTATS PAR FICHIER ===\n")
        for result in all_results:
            f.write(f"\n{result['file']}: {result['status']}\n")
            if result['status'] == 'success':
                metrics = result['results']['metrics']
                f.write(f"  Accuracy: {metrics['accuracy']*100:.1f}%\n")
                f.write(f"  Recall Cancer: {metrics['recall_per_class'][1]*100:.1f}%\n")
            else:
                f.write(f"  Erreur: {result.get('error', 'Inconnue')}\n")
        
        if global_stats['global_metrics']:
            gm = global_stats['global_metrics']
            f.write(f"\n=== PERFORMANCE GLOBALE ===\n")
            f.write(f"Total patches: {global_stats['total_patches']}\n")
            f.write(f"Accuracy globale: {gm['accuracy']*100:.1f}%\n")
            f.write(f"F1-Score global: {gm['f1_score']*100:.1f}%\n")
            f.write(f"Recall Cancer: {gm['recall_per_class'][1]*100:.1f}%\n")
            f.write(f"Precision Cancer: {gm['precision_per_class'][1]*100:.1f}%\n")
            
            tn, fp, fn, tp = gm['confusion_matrix'].ravel()
            f.write(f"\nMatrice de confusion globale:\n")
            f.write(f"VP (Cancer détecté): {tp}\n")
            f.write(f"VN (Sain détecté): {tn}\n")
            f.write(f"FP (Fausse alerte): {fp}\n")
            f.write(f"FN (Cancer raté): {fn}\n")
    
    # CSV des résultats
    df_results = pd.DataFrame([
        {
            'file': r['file'],
            'status': r['status'],
            'accuracy': r['results']['metrics']['accuracy'] if r['status'] == 'success' else None,
            'f1_score': r['results']['metrics']['f1_score'] if r['status'] == 'success' else None,
            'recall_cancer': r['results']['metrics']['recall_per_class'][1] if r['status'] == 'success' else None,
            'precision_cancer': r['results']['metrics']['precision_per_class'][1] if r['status'] == 'success' else None,
            'patches_processed': r['results']['processing_info']['processed_patches'] if r['status'] == 'success' else None,
            'error': r.get('error', '') if r['status'] == 'failed' else ''
        }
        for r in all_results
    ])
    
    df_results.to_csv(f"{save_dir}/batch_results_summary.csv", index=False)
    
    print(f"💾 Rapport batch sauvegardé dans: {save_dir}")

# Exécuter le test
if __name__ == "__main__":
    # Test sur un seul fichier WSI
    print("Choisissez le type de test:")
    print("1. Test sur un seul fichier WSI")
    print("2. Test batch sur plusieurs fichiers WSI")
    
    choice = input("Votre choix (1 ou 2): ").strip()
    
    if choice == "1":
        results, pipeline = run_wsi_test()
    elif choice == "2":
        all_results, global_stats, pipeline = run_batch_wsi_test()
    else:
        print("Choix invalide. Exécution du test simple...")
        results, pipeline = run_wsi_test() not annotations:
            print("❌ Aucune annotation trouvée dans le XML!")
            return []
        
        # Traiter toutes les annotations pour extraire les patches
        patches_data = []
        
        for idx, annotation in enumerate(annotations):
            print(f"   📍 Traitement annotation {idx+1}/{len(annotations)}: {annotation['name']}")
            
            # Extraire la région avec padding
            region_img, region_bounds = extractor.extract_annotation_region(annotation, padding)
            
            # Créer le masque pour cette région
            mask = extractor.create_mask_for_annotation(annotation, region_bounds)
            
            # Extraire les patches de cette région
            patches_cancer, patches_sain, patches_info = extractor.extract_patches_from_region(
                region_img, mask, region_bounds
            )
            
            # Combiner tous les patches (cancer + sain)
            all_patches = patches_cancer + patches_sain
            
            # Ajouter les patches à notre liste avec leurs vraies étiquettes
            for i, patch in enumerate(all_patches):
                # Déterminer si c'est un patch cancer ou sain
                is_cancer_patch = i < len(patches_cancer)
                
                patch_data = {
                    'patch': patch,
                    'true_label': 1 if is_cancer_patch else 0,  # 1=cancer, 0=sain
                    'annotation_idx': idx,
                    'annotation_name': annotation['name'],
                    'patch_idx': i,
                    'cancer_percentage': patches_info[i]['cancer_percentage'] if i < len(patches_info) else 0.0
                }
                patches_data.append(patch_data)
            
            print(f"      ✅ {len(patches_cancer)} patches cancéreux, {len(patches_sain)} patches sains")
        
        print(f"✅ Total patches extraits: {len(patches_data)}")
        print(f"   - Cancer: {sum(1 for p in patches_data if p['true_label'] == 1)}")
        print(f"   - Sain: {sum(1 for p in patches_data if p['true_label'] == 0)}")
        
        return patches_data
    
    def extract_embedding_from_patch(self, patch_array):
        """
        Extraire l'embedding d'un patch avec CTransPath
        
        Args:
            patch_array: Array numpy du patch (H, W, 3)
        
        Returns:
            embedding: Tensor des features
        """
        try:
            # Convertir en PIL Image
            if patch_array.dtype != np.uint8:
                patch_array = (patch_array * 255).astype(np.uint8)
            
            image = Image.fromarray(patch_array)
            
            # Appliquer les transformations
            tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Extraire l'embedding
            with torch.no_grad():
                embedding = self.ctrans_net(tensor).squeeze(0)
            
            return embedding
        except Exception as e:
            print(f"❌ Erreur lors de l'extraction d'embedding: {e}")
            return None
    
    def predict_patch_embedding(self, embedding):
        """
        Prédire la classe d'un patch à partir de son embedding
        
        Args:
            embedding: Tensor des features
        
        Returns:
            prediction: 0 (sain) ou 1 (cancer)
            confidence: Score de confiance [0-1]
            probabilities: [prob_sain, prob_cancer]
        """
        try:
            with torch.no_grad():
                embedding_batch = embedding.unsqueeze(0).to(self.device)
                logits = self.mlp_model(embedding_batch)
                probabilities = torch.softmax(logits, dim=1)
                
                pred_class = logits.argmax(dim=1).item()
                confidence = probabilities.max().item()
                prob_cancer = probabilities[0, 1].item()
                prob_sain = probabilities[0, 0].item()
                
                return pred_class, confidence, [prob_sain, prob_cancer]
        except Exception as e:
            print(f"❌ Erreur lors de la prédiction: {e}")
            return -1, 0.0, [0.0, 0.0]
    
    def test_wsi_complete(self, wsi_path, xml_path, save_dir="wsi_test_results", 
                         patch_size=256, padding=200):
        """
        Pipeline complet de test sur WSI
        
        Args:
            wsi_path: Chemin vers le fichier WSI
            xml_path: Chemin vers le fichier XML
            save_dir: Dossier pour sauvegarder les résultats
            patch_size: Taille des patches
            padding: Padding autour de la bbox
        
        Returns:
            results: Dictionnaire avec tous les résultats
        """
        os.makedirs(save_dir, exist_ok=True)
        
        print("🚀 DÉBUT DU TEST WSI")
        print("="*60)
        
        # 1. Extraire les patches depuis la WSI
        patches_data = self.extract_patches_from_wsi(wsi_path, xml_path, patch_size, padding)
        
        if not patches_data:
            print("❌ Aucun patch extrait!")
            return None
        
        # 2. Extraire les embeddings et faire les prédictions
        print(f"\n🔄 Extraction des embeddings et prédictions...")
        
        true_labels = []
        predictions = []
        confidences = []
        all_probabilities = []
        detailed_results = []
        failed_patches = 0
        
        start_time = time.time()
        
        for i, patch_data in enumerate(tqdm(patches_data, desc="Processing patches")):
            # Extraire l'embedding
            embedding = self.extract_embedding_from_patch(patch_data['patch'])
            
            if embedding is not None:
                # Faire la prédiction
                pred, conf, probs = self.predict_patch_embedding(embedding)
                
                if pred != -1:  # Prédiction valide
                    true_labels.append(patch_data['true_label'])
                    predictions.append(pred)
                    confidences.append(conf)
                    all_probabilities.append(probs)
                    
                    # Ajouter aux résultats détaillés
                    detailed_results.append({
                        'patch_idx': i,
                        'annotation_name': patch_data['annotation_name'],
                        'true_label': patch_data['true_label'],
                        'predicted_label': pred,
                        'confidence': conf,
                        'prob_sain': probs[0],
                        'prob_cancer': probs[1],
                        'correct': patch_data['true_label'] == pred,
                        'cancer_percentage': patch_data['cancer_percentage']
                    })
                else:
                    failed_patches += 1
            else:
                failed_patches += 1
        
        processing_time = time.time() - start_time
        
        if not true_labels:
            print("❌ Aucune prédiction valide!")
            return None
        
        # 3. Calculer les métriques
        print(f"\n📊 Calcul des métriques...")
        metrics = self.calculate_wsi_metrics(true_labels, predictions, all_probabilities)
        
        # 4. Créer les visualisations
        print(f"\n📈 Génération des visualisations...")
        self.create_wsi_visualizations(true_labels, predictions, all_probabilities, 
                                     detailed_results, metrics, save_dir)
        
        # 5. Sauvegarder les résultats
        self.save_wsi_results(detailed_results, metrics, processing_time, 
                            len(patches_data), failed_patches, save_dir)
        
        # 6. Résumé final
        print(f"\n🎉 TEST WSI TERMINÉ!")
        print(f"📊 Patches traités: {len(true_labels)}/{len(patches_data)}")
        print(f"❌ Patches échoués: {failed_patches}")
        print(f"⏱️ Temps de traitement: {processing_time:.2f}s")
        print(f"🎯 Accuracy: {metrics['accuracy']:.4f}")
        print(f"📈 F1-Score: {metrics['f1_score']:.4f}")
        print(f"🔍 Recall Cancer: {metrics['recall_per_class'][1]:.4f}")
        print(f"🎯 Precision Cancer: {metrics['precision_per_class'][1]:.4f}")
        
        results = {
            'metrics': metrics,
            'detailed_results': detailed_results,
            'processing_info': {
                'total_patches': len(patches_data),
                'processed_patches': len(true_labels),
                'failed_patches': failed_patches,
                'processing_time': processing_time,
                'wsi_path': wsi_path,
                'xml_path': xml_path
            }
        }
        
        return results
    
    def calculate_wsi_metrics(self, y_true, y_pred, probabilities):
        """
        Calculer toutes les métriques de performance pour WSI
        """
        # Métriques de base
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        
        # Métriques par classe
        f1_per_class = f1_score(y_true, y_pred, average=None)
        precision_per_class = precision_score(y_true, y_pred, average=None)
        recall_per_class = recall_score(y_true, y_pred, average=None)
        
        # Matrice de confusion
        cm = confusion_matrix(y_true, y_pred)
        
        # Rapport de classification
        class_report = classification_report(y_true, y_pred, 
                                           target_names=['Sain', 'Cancer'], 
                                           output_dict=True)
        
        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'f1_per_class': f1_per_class,
            'precision_per_class': precision_per_class,
            'recall_per_class': recall_per_class,
            'confusion_matrix': cm,
            'classification_report': class_report,
            'probabilities': probabilities
        }
    
    def create_wsi_visualizations(self, y_true, y_pred, probabilities, detailed_results, 
                                metrics, save_dir):
        """
        Créer toutes les visualisations pour le test WSI
        """
        # Configuration
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Matrice de confusion détaillée
        plt.figure(figsize=(12, 10))
        sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Sain', 'Cancer'], yticklabels=['Sain', 'Cancer'],
                   cbar_kws={'label': 'Nombre de patches'})
        plt.title('Matrice de Confusion - Test WSI', fontsize=16, fontweight='bold')
        plt.xlabel('Prédictions', fontsize=12)
        plt.ylabel('Valeurs Réelles', fontsize=12)
        
        # Ajouter des statistiques détaillées
        tn, fp, fn, tp = metrics['confusion_matrix'].ravel()
        stats_text = f'''MÉTRIQUES DÉTAILLÉES:
        
Accuracy: {metrics["accuracy"]:.3f}
F1-Score: {metrics["f1_score"]:.3f}
Precision: {metrics["precision"]:.3f}
Recall: {metrics["recall"]:.3f}

PAR CLASSE:
Cancer - Precision: {metrics["precision_per_class"][1]:.3f}
Cancer - Recall: {metrics["recall_per_class"][1]:.3f}
Cancer - F1: {metrics["f1_per_class"][1]:.3f}

CONFUSION:
VP (Cancer détecté): {tp}
VN (Sain détecté): {tn}
FP (Fausse alerte): {fp}
FN (Cancer raté): {fn}'''
        
        plt.figtext(0.02, 0.02, stats_text, fontsize=9, 
                   bbox=dict(boxstyle="round", facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/confusion_matrix_wsi.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Métriques par classe (style médical)
        classes = ['Tissus Sains', 'Tissus Cancéreux']
        x_pos = np.arange(len(classes))
        
        fig, ax = plt.subplots(figsize=(14, 8))
        width = 0.25
        
        bars1 = ax.bar(x_pos - width, metrics['precision_per_class'], width, 
                      label='Precision', alpha=0.8, color='skyblue')
        bars2 = ax.bar(x_pos, metrics['recall_per_class'], width, 
                      label='Recall (Sensibilité)', alpha=0.8, color='lightgreen')
        bars3 = ax.bar(x_pos + width, metrics['f1_per_class'], width, 
                      label='F1-Score', alpha=0.8, color='lightcoral')
        
        ax.set_xlabel('Types de Tissus', fontsize=12)
        ax.set_ylabel('Score de Performance', fontsize=12)
        ax.set_title('Performance du Modèle par Type de Tissu - Test WSI', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(classes)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Ajouter valeurs critiques
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                color = 'red' if height < 0.90 else 'green'
                weight = 'bold' if height < 0.90 else 'normal'
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom', 
                       fontweight=weight, color=color)
        
        # Ligne de référence pour performance acceptable
        ax.axhline(y=0.90, color='red', linestyle='--', alpha=0.7, 
                  label='Seuil Acceptable (90%)')
        ax.legend()
        
        plt.ylim(0, 1.1)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/metrics_per_class_wsi.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # 3. Distribution des probabilités cancer avec analyse critique
        cancer_probs = [prob[1] for prob in probabilities]
        
        plt.figure(figsize=(14, 8))
        
        # Séparer par vraie classe
        sain_probs = [cancer_probs[i] for i in range(len(y_true)) if y_true[i] == 0]
        cancer_true_probs = [cancer_probs[i] for i in range(len(y_true)) if y_true[i] == 1]
        
        plt.hist(sain_probs, bins=30, alpha=0.7, label=f'Vrais Tissus Sains (n={len(sain_probs)})', 
                color='lightblue', density=True)
        plt.hist(cancer_true_probs, bins=30, alpha=0.7, label=f'Vrais Cancers (n={len(cancer_true_probs)})', 
                color='lightcoral', density=True)
        
        plt.axvline(x=0.5, color='red', linestyle='--', linewidth=2, 
                   label='Seuil de décision (0.5)')
        plt.xlabel('Probabilité de Cancer prédite', fontsize=12)
        plt.ylabel('Densité de probabilité', fontsize=12)
        plt.title('Distribution des Probabilités Cancer - Analyse Diagnostique', 
                 fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
        
        # Ajouter des statistiques
        overlap_zone = len([p for p in sain_probs if p > 0.3]) + len([p for p in cancer_true_probs if p < 0.7])
        plt.figtext(0.02, 0.98, f'Zone d\'incertitude (0.3-0.7): {overlap_zone} patches', 
                   fontsize=10, bbox=dict(boxstyle="round", facecolor='yellow', alpha=0.7))
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/probability_distribution_wsi.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # 4. Analyse des erreurs critiques
        errors = [(i, y_true[i], y_pred[i], cancer_probs[i]) 
                 for i in range(len(y_true)) if y_true[i] != y_pred[i]]
        
        if errors:
            plt.figure(figsize=(14, 8))
            error_indices, true_labels, pred_labels, error_probs = zip(*errors)
            
            # Distinguer faux positifs et faux négatifs
            false_positives = [(i, prob) for i, (true, pred, prob) in enumerate(zip(true_labels, pred_labels, error_probs)) if true == 0 and pred == 1]
            false_negatives = [(i, prob) for i, (true, pred, prob) in enumerate(zip(true_labels, pred_labels, error_probs)) if true == 1 and pred == 0]
            
            if false_positives:
                fp_indices, fp_probs = zip(*false_positives)
                plt.scatter(fp_indices, fp_probs, c='red', alpha=0.8, s=100, 
                           label=f'Faux Positifs (n={len(false_positives)})', marker='x')
            
            if false_negatives:
                fn_indices, fn_probs = zip(*false_negatives)
                plt.scatter(fn_indices, fn_probs, c='orange', alpha=0.8, s=100, 
                           label=f'Faux Négatifs (n={len(false_negatives)})', marker='o')
            
            plt.axhline(y=0.5, color='black', linestyle='--', label='Seuil décision')
            plt.xlabel('Index des erreurs', fontsize=12)
            plt.ylabel('Probabilité Cancer prédite', fontsize=12)
            plt.title(f'Analyse des Erreurs Critiques ({len(errors)} erreurs sur {len(y_true)} patches)', 
                     fontsize=14, fontweight='bold')
            plt.legend()
            plt.grid(alpha=0.3)
            
            # Zones critiques
            plt.axhspan(0.4, 0.6, alpha=0.2, color='yellow', label='Zone d\'incertitude')
            
            plt.tight_layout()
            plt.savefig(f"{save_dir}/error_analysis_wsi.png", dpi=300, bbox_inches='tight')
            plt.show()
    
    def save_wsi_results(self, detailed_results, metrics, processing_time, 
                        total_patches, failed_patches, save_dir):
        """
        Sauvegarder tous les résultats du test WSI
        """
        # DataFrame des résultats détaillés
        df = pd.DataFrame(detailed_results)
        df.to_csv(f"{save_dir}/detailed_wsi_predictions.csv", index=False)
        
        # Rapport médical détaillé
        with open(f"{save_dir}/wsi_medical_report.txt", 'w', encoding='utf-8') as f:
            f.write("=== RAPPORT D'ANALYSE WSI - DÉTECTION CANCER ===\n\n")
            f.write(f"Date d'analyse: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Temps de traitement: {processing_time:.2f}s\n")
            f.write(f"Patches analysés: {len(detailed_results)}/{total_patches}\n")
            f.write(f"Patches échoués: {failed_patches}\n\n")
            
            f.write("=== PERFORMANCE GLOBALE ===\n")
            f.write(f"Accuracy (Précision globale): {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.1f}%)\n")
            f.write(f"F1-Score: {metrics['f1_score']:.4f}\n")
            f.write(f"Precision: {metrics['precision']:.4f}\n")
            f.write(f"Recall (Sensibilité): {metrics['recall']:.4f}\n\n")
            
            f.write("=== PERFORMANCE PAR TYPE DE TISSU ===\n")
            f.write("TISSUS SAINS:\n")
            f.write(f"  Precision: {metrics['precision_per_class'][0]:.4f}\n")
            f.write(f"  Recall: {metrics['recall_per_class'][0]:.4f}\n")
            f.write(f"  F1-Score: {metrics['f1_per_class'][0]:.4f}\n\n")
            
            f.write("TISSUS CANCÉREUX (CRITIQUE):\n")
            f.write(f"  Precision: {metrics['precision_per_class'][1]:.4f}\n")
            f.write(f"  Recall: {metrics['recall_per_class'][1]:.4f} ⚠️\n")
            f.write(f"  F1-Score: {metrics['f1_per_class'][1]:.4f}\n\n")
            
            f.write("=== ANALYSE CRITIQUE ===\n")
            tn, fp, fn, tp = metrics['confusion_matrix'].ravel()
            f.write(f"Vrais Positifs (Cancer détecté): {tp}\n")
            f.write(f"Vrais Négatifs (Sain détecté): {tn}\n")
            f.write(f"Faux Positifs (Fausse alerte cancer): {fp}\n")
            f.write(f"Faux Négatifs (Cancer non détecté): {fn} ⚠️⚠️⚠️\n\n")
            
            # Évaluation clinique
            if metrics['recall_per_class'][1] >= 0.95:
                f.write("✅ ÉVALUATION: Performance acceptable pour usage clinique\n")
            elif metrics['recall_per_class'][1] >= 0.90:
                f.write("⚠️ ÉVALUATION: Performance correcte mais à surveiller\n")
            else:
                f.write("❌ ÉVALUATION: Performance insuffisante pour usage clinique\n")
            
            f.write(f"\nMatrice de confusion:\n{metrics['confusion_matrix']}\n")
        
        print(f"💾 Résultats détaillés sauvegardés dans: {save_dir}")

def run_wsi_test():
    """
    Fonction principale pour tester le modèle sur WSI
    """
    # Chemins vers vos modèles
    MLP_MODEL_PATH = r'D:\Python code\results\mlp_model.pth'
    CTRANSPATH_MODEL_PATH = r'./ctranspath.pth'
    
    # Chemins vers vos données WSI de test
    WSI_PATH = r'd:/wsi/tumor_001.tif'  # Ajustez selon votre WSI de test
    XML_PATH = r'd:/wsi/annot/tumor_001.xml'  # Ajustez selon votre annotation
    
    # Paramètres d'extraction
    PATCH_SIZE = 256
    PADDING = 200
    
    # Dossier pour sauvegarder les résultats
    SAVE_DIR = './wsi_test_results'
    
    print("🏥 DÉBUT DU TEST WSI - DÉTECTION CANCER")
    print("="*60)
    print(f"📄 WSI: {WSI_PATH}")
    print(f"📄 XML: {XML_PATH}")
    print(f"🔬 Taille patches: {PATCH_SIZE}x{PATCH_SIZE}")
    print(f"📏 Padding: {PADDING}px")
    
    # Vérifier l'existence des fichiers
    if not os.path.exists(WSI_PATH):
        print(f"❌ Fichier WSI introuvable: {WSI_PATH}")
        return None
    
    if not os.path.exists(XML_PATH):
        print(f"❌ Fichier XML introuvable: {XML_PATH}")
        return None
    
    if not os.path.exists(MLP_MODEL_PATH):
        print(f"❌ Modèle MLP introuvable: {MLP_MODEL_PATH}")
        return None
    
    if not os.path.exists(CTRANSPATH_MODEL_PATH):
        print(f"❌ Modèle CTransPath introuvable: {CTRANSPATH_MODEL_PATH}")
        return None
    
    try:
        # Initialiser le pipeline de test
        pipeline = WSITestPipeline(MLP_MODEL_PATH, CTRANSPATH_MODEL_PATH)
        
        # Exécuter le test complet
        results = pipeline.test_wsi_complete(
            wsi_path=WSI_PATH,
            xml_path=XML_PATH,
            save_dir=SAVE_DIR,
            patch_size=PATCH_SIZE,
            padding=PADDING
        )
        
        if results is not None:
            print("\n🎉 TEST WSI TERMINÉ AVEC SUCCÈS!")
            print(f"📊 Résultats sauvegardés dans: {SAVE_DIR}")
            
            # Afficher un résumé final
            metrics = results['metrics']
            print("\n📋 RÉSUMÉ FINAL:")
            print(f"   🎯 Accuracy: {metrics['accuracy']:.3f}")
            print(f"   📈 F1-Score: {metrics['f1_score']:.3f}")
            print(f"   🔍 Recall Cancer: {metrics['recall_per_class'][1]:.3f}")
            print(f"   🎯 Precision Cancer: {metrics['precision_per_class'][1]:.3f}")
            
            # Évaluation critique
            recall_cancer = metrics['recall_per_class'][1]
            if recall_cancer >= 0.95:
                print("   ✅ Performance: Excellent pour usage clinique")
            elif recall_cancer >= 0.90:
                print("   ⚠️ Performance: Correct mais à surveiller")
            else:
                print("   ❌ Performance: Insuffisant pour usage clinique")
            
            return results
        else:
            print("❌ Le test WSI a échoué!")
            return None
            
    except Exception as e:
        print(f"❌ Erreur lors du test WSI: {e}")
        print("💡 Vérifiez que tous les fichiers existent et que les modèles sont compatibles")
        return None

def run_batch_wsi_test(wsi_xml_pairs, mlp_model_path, ctranspath_model_path, 
                       base_save_dir="batch_wsi_results"):
    """
    Fonction pour tester le modèle sur plusieurs WSI
    
    Args:
        wsi_xml_pairs: Liste de tuples (wsi_path, xml_path)
        mlp_model_path: Chemin vers le modèle MLP
        ctranspath_model_path: Chemin vers le modèle CTransPath
        base_save_dir: Dossier de base pour les résultats
    
    Returns:
        batch_results: Dictionnaire avec tous les résultats
    """
    print("🏥 DÉBUT DU TEST BATCH WSI")
    print("="*60)
    print(f"📊 Nombre de WSI à tester: {len(wsi_xml_pairs)}")
    
    # Créer le dossier de base
    os.makedirs(base_save_dir, exist_ok=True)
    
    # Initialiser le pipeline une seule fois
    try:
        pipeline = WSITestPipeline(mlp_model_path, ctranspath_model_path)
    except Exception as e:
        print(f"❌ Erreur lors de l'initialisation: {e}")
        return None
    
    batch_results = {}
    all_metrics = []
    total_patches = 0
    total_processing_time = 0
    
    for i, (wsi_path, xml_path) in enumerate(wsi_xml_pairs):
        print(f"\n🔄 Test WSI {i+1}/{len(wsi_xml_pairs)}")
        print(f"   WSI: {os.path.basename(wsi_path)}")
        print(f"   XML: {os.path.basename(xml_path)}")
        
        # Créer un sous-dossier pour chaque WSI
        wsi_name = os.path.splitext(os.path.basename(wsi_path))[0]
        save_dir = os.path.join(base_save_dir, f"wsi_{i+1}_{wsi_name}")
        
        try:
            # Tester cette WSI
            results = pipeline.test_wsi_complete(
                wsi_path=wsi_path,
                xml_path=xml_path,
                save_dir=save_dir
            )
            
            if results is not None:
                batch_results[wsi_name] = results
                all_metrics.append(results['metrics'])
                total_patches += results['processing_info']['processed_patches']
                total_processing_time += results['processing_info']['processing_time']
                
                print(f"   ✅ Succès - Accuracy: {results['metrics']['accuracy']:.3f}")
            else:
                print(f"   ❌ Échec pour {wsi_name}")
                
        except Exception as e:
            print(f"   ❌ Erreur pour {wsi_name}: {e}")
            continue
    
    # Calculer les statistiques globales
    if all_metrics:
        print(f"\n📊 STATISTIQUES GLOBALES DU BATCH:")
        print(f"   WSI testées avec succès: {len(all_metrics)}/{len(wsi_xml_pairs)}")
        print(f"   Total patches traités: {total_patches}")
        print(f"   Temps total: {total_processing_time:.2f}s")
        
        # Moyennes des métriques
        avg_accuracy = np.mean([m['accuracy'] for m in all_metrics])
        avg_f1 = np.mean([m['f1_score'] for m in all_metrics])
        avg_recall_cancer = np.mean([m['recall_per_class'][1] for m in all_metrics])
        avg_precision_cancer = np.mean([m['precision_per_class'][1] for m in all_metrics])
        
        print(f"   Accuracy moyenne: {avg_accuracy:.3f}")
        print(f"   F1-Score moyen: {avg_f1:.3f}")
        print(f"   Recall Cancer moyen: {avg_recall_cancer:.3f}")
        print(f"   Precision Cancer moyenne: {avg_precision_cancer:.3f}")
        
        # Sauvegarder le rapport global
        save_batch_summary(batch_results, all_metrics, base_save_dir)
    
    return batch_results

def save_batch_summary(batch_results, all_metrics, save_dir):
    """
    Sauvegarder le résumé du test batch
    """
    with open(f"{save_dir}/batch_summary.txt", 'w', encoding='utf-8') as f:
        f.write("=== RÉSUMÉ DU TEST BATCH WSI ===\n\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Nombre de WSI testées: {len(batch_results)}\n\n")
        
        f.write("=== RÉSULTATS PAR WSI ===\n")
        for wsi_name, results in batch_results.items():
            metrics = results['metrics']
            f.write(f"\n{wsi_name}:\n")
            f.write(f"  Accuracy: {metrics['accuracy']:.4f}\n")
            f.write(f"  F1-Score: {metrics['f1_score']:.4f}\n")
            f.write(f"  Recall Cancer: {metrics['recall_per_class'][1]:.4f}\n")
            f.write(f"  Precision Cancer: {metrics['precision_per_class'][1]:.4f}\n")
            f.write(f"  Patches traités: {results['processing_info']['processed_patches']}\n")
        
        f.write("\n=== STATISTIQUES GLOBALES ===\n")
        avg_accuracy = np.mean([m['accuracy'] for m in all_metrics])
        avg_f1 = np.mean([m['f1_score'] for m in all_metrics])
        avg_recall_cancer = np.mean([m['recall_per_class'][1] for m in all_metrics])
        
        f.write(f"Accuracy moyenne: {avg_accuracy:.4f}\n")
        f.write(f"F1-Score moyen: {avg_f1:.4f}\n")
        f.write(f"Recall Cancer moyen: {avg_recall_cancer:.4f}\n")
        
        # Évaluation globale
        if avg_recall_cancer >= 0.95:
            f.write("\n✅ ÉVALUATION GLOBALE: Performance acceptable\n")
        elif avg_recall_cancer >= 0.90:
            f.write("\n⚠️ ÉVALUATION GLOBALE: Performance à surveiller\n")
        else:
            f.write("\n❌ ÉVALUATION GLOBALE: Performance insuffisante\n")

if __name__ == "__main__":
    print("🚀 LANCEMENT DU PIPELINE DE TEST WSI")
    print("Choisissez le mode de test:")
    print("1. Test d'une seule WSI")
    print("2. Test batch (plusieurs WSI)")
    
    choice = input("Votre choix (1 ou 2): ").strip()
    
    if choice == "1":
        # Test d'une seule WSI
        results = run_wsi_test()
        if results:
            print("✅ Test terminé avec succès!")
        else:
            print("❌ Test échoué!")
            
    elif choice == "2":
        # Test batch - Exemple avec plusieurs WSI
        wsi_xml_pairs = [
            (r'd:/wsi/tumor_001.tif', r'd:/wsi/annot/tumor_001.xml'),
            (r'd:/wsi/tumor_002.tif', r'd:/wsi/annot/tumor_002.xml'),
            # Ajoutez d'autres paires selon vos données
        ]
        
        MLP_MODEL_PATH = r'D:\Python code\results\mlp_model.pth'
        CTRANSPATH_MODEL_PATH = r'./ctranspath.pth'
        
        batch_results = run_batch_wsi_test(
            wsi_xml_pairs=wsi_xml_pairs,
            mlp_model_path=MLP_MODEL_PATH,
            ctranspath_model_path=CTRANSPATH_MODEL_PATH
        )
        
        if batch_results:
            print("✅ Test batch terminé!")
        else:
            print("❌ Test batch échoué!")
    
    else:
        print("❌ Choix invalide!")