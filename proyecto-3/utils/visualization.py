import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import os
from PIL import Image

def plot_distribution(dataset_structure):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    splits = list(dataset_structure.keys())
    counts = [sum(split_counts.values()) for split_counts in dataset_structure.values()]
    
    bars = ax.bar(splits, counts, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax.set_title('Distribución de imágenes por conjunto')
    ax.set_ylabel('Número de imágenes')
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

def plot_samples_per_species(dataset_structure, split='train'):
    species_counts = dataset_structure[split]
    sorted_species = sorted(species_counts.items(), key=lambda x: x[1], reverse=True) 
    
    species_names, counts = zip(*sorted_species)
    
    plt.figure(figsize=(12, 6))
    plt.barh(species_names, counts, color='skyblue')
    plt.title(f'Distribución de imágenes por especie ({split.upper()})', fontsize=14, pad=15)
    plt.xlabel('Número de imágenes')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

def display_sample_images(dataset, n_samples=3):
    samples_by_class = defaultdict(list)
    
    for img_path, label in dataset.samples:
        if len(samples_by_class[dataset.classes[label]]) < n_samples:
            samples_by_class[dataset.classes[label]].append(img_path)
    
    fig, axes = plt.subplots(len(dataset.classes), n_samples, figsize=(15, 30))
    
    for i, cls in enumerate(dataset.classes):
        for j in range(n_samples):
            if j < len(samples_by_class[cls]):
                img = Image.open(samples_by_class[cls][j])
                axes[i, j].imshow(img)
                axes[i, j].set_title(f"{cls}" if j == 0 else "")
                axes[i, j].axis('off')
            else:
                axes[i, j].axis('off')
    
    plt.tight_layout()
    plt.show()