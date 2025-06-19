import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from time import time
import os
from collections import defaultdict

output_folder = './tsne_visualizations2/'
os.makedirs(output_folder, exist_ok=True) 


def plot_embedding(X_emb, labels, epoch):
    
    x_min, x_max = np.min(X_emb, 0), np.max(X_emb, 0)
    X_emb = (X_emb - x_min) / (x_max - x_min)

    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111)
    
    unique_labels = np.unique(labels)  
    colors = plt.cm.get_cmap("Set1", len(unique_labels))  
    highlight_ratio = 0.05 

    
    highlight_indices = defaultdict(list)
    for c in unique_labels:
        class_indices = np.where(labels == c)[0] 
        n_highlight = max(1, int(len(class_indices) * highlight_ratio)) 
        highlight_indices[c] = np.random.choice(class_indices, n_highlight, replace=False)

    
    for i, c in enumerate(unique_labels):
        class_indices = np.where(labels == c)[0]
        ax.scatter(X_emb[class_indices, 0], X_emb[class_indices, 1],
                   color=colors(i), alpha=0.2, s=50)  

    
    for i, c in enumerate(unique_labels):
        high_indices = highlight_indices[c]
        ax.scatter(X_emb[high_indices, 0], X_emb[high_indices, 1],
                   color=colors(i), edgecolors='black', s=100, alpha=1.0, linewidth=1.5)  

    plt.xticks([]), plt.yticks([])
    save_name = os.path.join(output_folder, f"tsne_epoch_{epoch}.png")
    plt.savefig(save_name)
    plt.close()  
    print(f"t-SNE visualization saved to {save_name}")


def load_adversarial_samples(file_path):
    
    checkpoint = torch.load(file_path)
    samples = checkpoint['samples']  
    targets = checkpoint['targets']
    
    
    samples = samples.view(samples.size(0), -1)
    
    return samples.numpy(), targets.numpy()


def visualize_adversarial_samples(file_path, epoch):
   
    print(f"Loading adversarial samples from: {file_path}")
    X_adv, y_adv = load_adversarial_samples(file_path)

    X_adv = (X_adv - X_adv.min()) / (X_adv.max() - X_adv.min())

    
    print("Computing t-SNE embedding")
    t0 = time()
    X_tsne = TSNE(n_components=2, init='random', random_state=42, perplexity=30).fit_transform(X_adv)

    
    plot_embedding(X_tsne, y_adv, epoch)
    print("t-SNE completed in %.2f seconds" % (time() - t0))


for epoch in range(0, 200, 20):  
    file_path = f'./adversarial_samples/adv_samples_epoch_{epoch}.pt'
    if os.path.exists(file_path):
        visualize_adversarial_samples(file_path, epoch)
    else:
        print(f"File not found: {file_path}")
