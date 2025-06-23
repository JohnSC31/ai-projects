import torch
import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision.utils import make_grid

def extract_latent_vectors(model, dataloader, device='cuda'):
    """Extrae vectores latentes para todo el dataset"""
    model.eval()
    latent_vectors = []
    labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            x, y = batch
            x = x.to(device)
            z = model(x, return_latent=True)
            latent_vectors.append(z.cpu())
            labels.append(y)
    
    return torch.cat(latent_vectors), torch.cat(labels)

def visualize_tsne(latent_vectors, labels, save_path='latent_space.png'):
    """Visualización 2D del espacio latente"""
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    latent_2d = tsne.fit_transform(latent_vectors)
    
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x=latent_2d[:, 0], y=latent_2d[:, 1],
                    hue=labels, palette="husl", alpha=0.7)
    plt.title("t-SNE del Espacio Latente")
    plt.savefig(save_path)
    plt.close()
    return latent_2d

def cluster_analysis(latent_vectors, n_clusters=30):
    """Análisis de clusters con K-Means"""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(latent_vectors)
    print(f"Silhouette Score: {silhouette_score(latent_vectors, clusters):.4f}")
    return clusters

def visualize_clusters(dataset, clusters, n_samples=5, save_dir='clusters'):
    """Visualiza muestras de cada cluster"""
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    for cluster_id in np.unique(clusters):
        indices = np.where(clusters == cluster_id)[0][:n_samples]
        images = [dataset[i][0] for i in indices]
        
        grid = make_grid(images, nrow=n_samples)
        plt.figure(figsize=(15, 3))
        plt.imshow(grid.permute(1, 2, 0))
        plt.title(f"Cluster {cluster_id}")
        plt.axis('off')
        plt.savefig(f'{save_dir}/cluster_{cluster_id}.png')
        plt.close()