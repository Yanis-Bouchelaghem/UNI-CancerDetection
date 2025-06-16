import xml.etree.ElementTree as ET
import numpy as np
import cv2
from PIL import Image
import os
from pathlib import Path
import openslide
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon, Rectangle
import matplotlib.patches as mpatches
import json
import warnings
import glob
import pandas as pd
from datetime import datetime
warnings.filterwarnings('ignore')

class WSICancerPatchExtractor:
    """
    Extracteur de patches depuis une WSI avec classification cancéreux/sain
    basée sur les annotations XML.
    """
    
    def __init__(self, wsi_path: str, xml_path: str, patch_size: int = 256):
        """
        Initialise l'extracteur.
        
        Args:
            wsi_path: Chemin vers le fichier TIFF de la WSI
            xml_path: Chemin vers le fichier XML d'annotations
            patch_size: Taille des patches (défaut: 256x256)
        """
        self.wsi_path = os.path.abspath(wsi_path)
        self.xml_path = os.path.abspath(xml_path)
        self.patch_size = patch_size
        self.annotations = []
        self.wsi = None
        self.use_openslide = False
        
        # Vérifier que les fichiers existent
        if not os.path.exists(self.wsi_path):
            raise FileNotFoundError(f"Fichier WSI non trouvé : {self.wsi_path}")
        if not os.path.exists(self.xml_path):
            raise FileNotFoundError(f"Fichier XML non trouvé : {self.xml_path}")
    
    def load_wsi(self):
        """Charge la WSI avec OpenSlide ou PIL."""
        try:
            self.wsi = openslide.OpenSlide(self.wsi_path)
            self.use_openslide = True
            print(f"✓ WSI chargée avec OpenSlide")
            print(f"  Dimensions : {self.wsi.dimensions}")
            print(f"  Niveaux : {self.wsi.level_count}")
        except Exception as e:
            print(f"OpenSlide non disponible : {e}")
            print("Chargement avec PIL...")
            self.wsi = Image.open(self.wsi_path)
            self.use_openslide = False
            print(f"✓ Image chargée avec PIL")
            print(f"  Dimensions : {self.wsi.size}")
    
    def parse_xml_annotations(self):
        """Parse le fichier XML et extrait les annotations de polygones."""
        tree = ET.parse(self.xml_path)
        root = tree.getroot()
        
        self.annotations = []
        
        for annotation in root.findall('.//Annotation'):
            if annotation.get('Type') in ['Polygon', 'Spline']:
                name = annotation.get('Name')
                color = annotation.get('Color')
                group = annotation.get('PartOfGroup')
                
                coordinates = []
                for coord in annotation.findall('.//Coordinate'):
                    x = float(coord.get('X'))
                    y = float(coord.get('Y'))
                    coordinates.append((x, y))
                
                if coordinates:
                    polygon = Polygon(coordinates)
                    self.annotations.append({
                        'name': name,
                        'color': color,
                        'group': group,
                        'coordinates': coordinates,
                        'polygon': polygon,
                        'bounds': polygon.bounds  # (minx, miny, maxx, maxy)
                    })
        
        print(f"✓ {len(self.annotations)} annotations trouvées dans le XML")
        return self.annotations
    
    def extract_annotation_region(self, annotation, padding=50):
        """
        Extrait la région contenant une annotation avec padding.
        
        Args:
            annotation: Dictionnaire d'annotation
            padding: Pixels de padding autour de la région
            
        Returns:
            region_image: Image numpy de la région
            region_bounds: (x, y, width, height) de la région extraite
        """
        minx, miny, maxx, maxy = annotation['bounds']
        
        # Ajouter le padding
        x = max(0, int(minx - padding))
        y = max(0, int(miny - padding))
        
        if self.use_openslide:
            w = min(self.wsi.dimensions[0] - x, int(maxx - minx + 2 * padding))
            h = min(self.wsi.dimensions[1] - y, int(maxy - miny + 2 * padding))
            region = self.wsi.read_region((x, y), 0, (w, h))
            region = region.convert('RGB')
        else:
            w = min(self.wsi.size[0] - x, int(maxx - minx + 2 * padding))
            h = min(self.wsi.size[1] - y, int(maxy - miny + 2 * padding))
            region = self.wsi.crop((x, y, x + w, y + h))
        
        return np.array(region), (x, y, w, h)
    
    def create_mask_for_annotation(self, annotation, region_bounds):
        """
        Crée un masque binaire pour l'annotation dans une région donnée.
        
        Args:
            annotation: Dictionnaire d'annotation
            region_bounds: (x, y, w, h) de la région
            
        Returns:
            mask: Masque binaire numpy
        """
        x, y, w, h = region_bounds
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Ajuster les coordonnées au repère de la région
        adjusted_coords = []
        for px, py in annotation['coordinates']:
            adj_x = int(px - x)
            adj_y = int(py - y)
            if 0 <= adj_x < w and 0 <= adj_y < h:
                adjusted_coords.append([adj_x, adj_y])
        
        if adjusted_coords:
            pts = np.array(adjusted_coords, dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)
        
        return mask
    
    def visualize_annotation_region(self, annotation, save_path=None):
        """
        Visualise une région annotée avec overlay semi-transparent.
        
        Args:
            annotation: Dictionnaire d'annotation
            save_path: Chemin pour sauvegarder la visualisation (optionnel)
        """
        # Extraire la région
        region_img, region_bounds = self.extract_annotation_region(annotation)
        x, y, w, h = region_bounds
        
        # Créer le masque
        mask = self.create_mask_for_annotation(annotation, region_bounds)
        
        # Créer l'overlay rouge semi-transparent
        overlay = region_img.copy()
        overlay[mask > 0] = [255, 0, 0]  # Rouge pour les zones cancéreuses
        
        # Mélanger l'image originale et l'overlay
        alpha = 0.3
        result = cv2.addWeighted(region_img, 1 - alpha, overlay, alpha, 0)
        
        # Dessiner le contour
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, (255, 0, 0), 3)
        
        # Afficher
        plt.figure(figsize=(12, 8))
        plt.imshow(result)
        plt.title(f"Annotation: {annotation['name']} (Groupe: {annotation['group']})")
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
        
        return result, mask
    
    def extract_patches_from_region(self, region_img, mask, region_bounds):
        """
        Découpe une région en patches et calcule le pourcentage de tissu cancéreux.
        
        Args:
            region_img: Image numpy de la région
            mask: Masque binaire de l'annotation
            region_bounds: (x, y, w, h) de la région dans l'image complète
            
        Returns:
            patches_cancer: Liste de patches cancéreux (>= 20% de tissu annoté)
            patches_sain: Liste de patches sains (< 20% de tissu annoté)
            patches_info: Informations sur chaque patch
        """
        x_offset, y_offset, _, _ = region_bounds
        patches_cancer = []
        patches_sain = []
        patches_info = []
        
        h, w = region_img.shape[:2]
        
        # Parcourir la région avec une fenêtre glissante
        for y in range(0, h - self.patch_size + 1, self.patch_size):
            for x in range(0, w - self.patch_size + 1, self.patch_size):
                # Extraire le patch
                patch = region_img[y:y + self.patch_size, x:x + self.patch_size]
                
                # Extraire la partie correspondante du masque
                patch_mask = mask[y:y + self.patch_size, x:x + self.patch_size]
                
                # Calculer le pourcentage de pixels cancéreux
                cancer_pixels = np.sum(patch_mask > 0)
                total_pixels = self.patch_size * self.patch_size
                cancer_percentage = (cancer_pixels / total_pixels) * 100
                
                # Position globale du patch dans la WSI
                global_x = x + x_offset
                global_y = y + y_offset
                
                # Convertir les types numpy en types Python standard pour JSON
                patch_info = {
                    'global_position': (int(global_x), int(global_y)),
                    'cancer_percentage': float(cancer_percentage),
                    'is_cancer': bool(cancer_percentage >= 20)  # Conversion explicite en bool Python
                }
                
                if cancer_percentage >= 20:
                    patches_cancer.append(patch)
                else:
                    patches_sain.append(patch)
                
                patches_info.append(patch_info)
        
        return patches_cancer, patches_sain, patches_info
    
    def process_all_annotations(self, output_dir):
        """
        Traite toutes les annotations et sauvegarde les patches.
        
        Args:
            output_dir: Répertoire de sortie principal
        """
        # Créer les dossiers de sortie
        output_dir = Path(output_dir)
        cancer_dir = output_dir / "tissu_cancereux"
        sain_dir = output_dir / "tissu_sain"
        viz_dir = output_dir / "visualisations"
        
        for dir_path in [cancer_dir, sain_dir, viz_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Statistiques globales
        total_patches_cancer = 0
        total_patches_sain = 0
        all_patches_info = []
        
        # Traiter chaque annotation
        print("\nTraitement des annotations...")
        for idx, annotation in enumerate(self.annotations):
            print(f"\n--- Annotation {idx + 1}/{len(self.annotations)}: {annotation['name']} ---")
            
            # Visualiser et sauvegarder la région annotée
            viz_path = viz_dir / f"annotation_{annotation['name']}_{idx}.png"
            result_img, mask = self.visualize_annotation_region(annotation, save_path=viz_path)
            
            # Extraire la région
            region_img, region_bounds = self.extract_annotation_region(annotation)
            
            # Recréer le masque pour cette région
            mask = self.create_mask_for_annotation(annotation, region_bounds)
            
            # Extraire les patches
            patches_cancer, patches_sain, patches_info = self.extract_patches_from_region(
                region_img, mask, region_bounds
            )
            
            # Sauvegarder les patches cancéreux
            for patch_idx, patch in enumerate(patches_cancer):
                filename = f"ann{idx}_{annotation['name']}_patch{patch_idx}_cancer.png"
                Image.fromarray(patch).save(cancer_dir / filename)
            
            # Sauvegarder les patches sains
            for patch_idx, patch in enumerate(patches_sain):
                filename = f"ann{idx}_{annotation['name']}_patch{patch_idx}_sain.png"
                Image.fromarray(patch).save(sain_dir / filename)
            
            # Ajouter les infos de cette annotation
            for info in patches_info:
                info['annotation_idx'] = idx
                info['annotation_name'] = annotation['name']
            all_patches_info.extend(patches_info)
            
            # Statistiques
            total_patches_cancer += len(patches_cancer)
            total_patches_sain += len(patches_sain)
            
            print(f"  Patches cancéreux: {len(patches_cancer)}")
            print(f"  Patches sains: {len(patches_sain)}")
        
        # Sauvegarder les métadonnées
        metadata = {
            'patch_size': self.patch_size,
            'total_annotations': len(self.annotations),
            'total_patches_cancer': total_patches_cancer,
            'total_patches_sain': total_patches_sain,
            'patches_info': all_patches_info
        }
        
        # Sauvegarder en JSON
        try:
            with open(output_dir / 'metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            print(f"Erreur lors de la sauvegarde des métadonnées : {e}")
        
        # Afficher le résumé
        print("\n=== RÉSUMÉ ===")
        print(f"Annotations traitées: {len(self.annotations)}")
        print(f"Total patches cancéreux: {total_patches_cancer}")
        print(f"Total patches sains: {total_patches_sain}")
        if total_patches_cancer + total_patches_sain > 0:
            ratio = total_patches_cancer/(total_patches_cancer + total_patches_sain)
            print(f"Ratio cancer/sain: {ratio:.2%}")
        
        return all_patches_info
    
    def create_patch_overview(self, patches_info, output_path, downsample=32):
        """
        Crée une vue d'ensemble montrant l'emplacement des patches cancéreux/sains.
        
        Args:
            patches_info: Liste des informations sur les patches
            output_path: Chemin de sauvegarde
            downsample: Facteur de réduction pour la visualisation
        """
        # Obtenir une miniature de la WSI
        if self.use_openslide:
            thumb_size = (self.wsi.dimensions[0] // downsample, 
                         self.wsi.dimensions[1] // downsample)
            thumbnail = self.wsi.get_thumbnail(thumb_size)
        else:
            thumb_size = (self.wsi.size[0] // downsample, 
                         self.wsi.size[1] // downsample)
            thumbnail = self.wsi.resize(thumb_size, Image.Resampling.LANCZOS)
        
        # Créer la figure
        fig, ax = plt.subplots(1, 1, figsize=(15, 15))
        ax.imshow(thumbnail)
        
        # Dessiner les patches
        for patch_info in patches_info:
            x, y = patch_info['global_position']
            x_scaled = x / downsample
            y_scaled = y / downsample
            size_scaled = self.patch_size / downsample
            
            # Couleur selon le type
            if patch_info['is_cancer']:
                color = 'red'
                alpha = 0.5
            else:
                color = 'green'
                alpha = 0.3
            
            rect = Rectangle((x_scaled, y_scaled), size_scaled, size_scaled,
                           linewidth=1, edgecolor=color, facecolor=color, alpha=alpha)
            ax.add_patch(rect)
        
        # Ajouter les contours des annotations
        for annotation in self.annotations:
            scaled_coords = [(x/downsample, y/downsample) 
                           for x, y in annotation['coordinates']]
            poly = MplPolygon(scaled_coords, fill=False, edgecolor='blue', 
                            linewidth=2, linestyle='--')
            ax.add_patch(poly)
        
        # Légende
        red_patch = mpatches.Patch(color='red', alpha=0.5, label='Tissu cancéreux (≥20%)')
        green_patch = mpatches.Patch(color='green', alpha=0.3, label='Tissu sain (<20%)')
        blue_line = mpatches.Patch(color='blue', label='Contour des annotations')
        ax.legend(handles=[red_patch, green_patch, blue_line], loc='upper right')
        
        ax.set_title(f"Vue d'ensemble des patches ({self.patch_size}x{self.patch_size} pixels)")
        ax.axis('off')
        
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Vue d'ensemble sauvegardée : {output_path}")


# === NOUVELLES FONCTIONS POUR LE TRAITEMENT EN BATCH ===

def process_single_wsi(wsi_path, xml_path, output_dir, patch_size=256):
    """
    Traite un seul fichier WSI et retourne les statistiques.
    
    Args:
        wsi_path: Chemin du fichier TIFF
        xml_path: Chemin du fichier XML
        output_dir: Dossier de sortie principal
        patch_size: Taille des patches
        
    Returns:
        dict: Statistiques du traitement
    """
    # Nom de base du fichier pour créer un sous-dossier
    base_name = Path(wsi_path).stem
    specific_output_dir = Path(output_dir) / base_name
    
    print(f"\n{'='*60}")
    print(f"Traitement de : {base_name}")
    print(f"{'='*60}")
    
    try:
        # Créer l'extracteur
        extractor = WSICancerPatchExtractor(wsi_path, xml_path, patch_size)
        
        # Charger la WSI
        extractor.load_wsi()
        
        # Parser les annotations
        extractor.parse_xml_annotations()
        
        # Traiter toutes les annotations
        patches_info = extractor.process_all_annotations(specific_output_dir)
        
        # Créer une vue d'ensemble
        if patches_info:
            overview_path = specific_output_dir / "vue_ensemble_patches.png"
            extractor.create_patch_overview(patches_info, overview_path)
        
        # Calculer les statistiques
        total_cancer = sum(1 for p in patches_info if p['is_cancer'])
        total_sain = sum(1 for p in patches_info if not p['is_cancer'])
        
        stats = {
            'file': base_name,
            'status': 'success',
            'annotations': len(extractor.annotations),
            'patches_cancer': total_cancer,
            'patches_sain': total_sain,
            'total_patches': total_cancer + total_sain,
            'ratio_cancer': total_cancer / (total_cancer + total_sain) if (total_cancer + total_sain) > 0 else 0,
            'output_dir': str(specific_output_dir)
        }
        
        print(f"✓ Traitement terminé pour {base_name}")
        
    except Exception as e:
        print(f"❌ Erreur lors du traitement de {base_name}: {e}")
        stats = {
            'file': base_name,
            'status': 'error',
            'error': str(e)
        }
    
    return stats


def process_batch_from_list(file_pairs, output_dir, patch_size=256):
    """
    Traite une liste de paires (wsi_path, xml_path).
    
    Args:
        file_pairs: Liste de tuples (wsi_path, xml_path)
        output_dir: Dossier de sortie principal
        patch_size: Taille des patches
    """
    print(f"\n=== TRAITEMENT EN BATCH DE {len(file_pairs)} FICHIERS ===\n")
    
    # Créer le dossier de sortie principal
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Statistiques globales
    all_stats = []
    
    # Traiter chaque fichier
    for i, (wsi_path, xml_path) in enumerate(file_pairs, 1):
        print(f"\nFichier {i}/{len(file_pairs)}")
        stats = process_single_wsi(wsi_path, xml_path, output_dir, patch_size)
        all_stats.append(stats)
    
    # Sauvegarder le rapport global
    report_path = output_dir / f"rapport_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, 'w') as f:
        json.dump(all_stats, f, indent=2)
    
    # Créer un rapport CSV
    df = pd.DataFrame(all_stats)
    csv_path = output_dir / f"rapport_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(csv_path, index=False)
    
    # Afficher le résumé
    print("\n" + "="*60)
    print("RÉSUMÉ DU TRAITEMENT BATCH")
    print("="*60)
    
    success_count = sum(1 for s in all_stats if s.get('status') == 'success')
    error_count = sum(1 for s in all_stats if s.get('status') == 'error')
    
    print(f"Fichiers traités avec succès : {success_count}/{len(file_pairs)}")
    print(f"Fichiers avec erreurs : {error_count}/{len(file_pairs)}")
    
    if success_count > 0:
        total_cancer = sum(s.get('patches_cancer', 0) for s in all_stats if s.get('status') == 'success')
        total_sain = sum(s.get('patches_sain', 0) for s in all_stats if s.get('status') == 'success')
        
        print(f"\nTotal patches cancéreux : {total_cancer}")
        print(f"Total patches sains : {total_sain}")
        print(f"Ratio global cancer/sain : {total_cancer/(total_cancer + total_sain):.2%}" if (total_cancer + total_sain) > 0 else "")
    
    print(f"\nRapports sauvegardés :")
    print(f"  - {report_path}")
    print(f"  - {csv_path}")


def process_batch_from_directory(wsi_dir, xml_dir, output_dir, patch_size=256):
    """
    Traite tous les fichiers .tif d'un dossier avec leurs XML correspondants.
    
    Args:
        wsi_dir: Dossier contenant les fichiers .tif
        xml_dir: Dossier contenant les fichiers .xml
        output_dir: Dossier de sortie
        patch_size: Taille des patches
    """
    # Trouver tous les fichiers .tif
    wsi_files = glob.glob(os.path.join(wsi_dir, "*.tif"))
    
    # Créer les paires de fichiers
    file_pairs = []
    missing_xml = []
    
    for wsi_path in wsi_files:
        base_name = Path(wsi_path).stem
        xml_path = os.path.join(xml_dir, f"{base_name}.xml")
        
        if os.path.exists(xml_path):
            file_pairs.append((wsi_path, xml_path))
        else:
            missing_xml.append(base_name)
    
    print(f"Fichiers trouvés : {len(wsi_files)}")
    print(f"Paires complètes : {len(file_pairs)}")
    
    if missing_xml:
        print(f"\n⚠️ Fichiers XML manquants pour : {', '.join(missing_xml)}")
    
    if file_pairs:
        process_batch_from_list(file_pairs, output_dir, patch_size)
    else:
        print("❌ Aucune paire de fichiers valide trouvée.")