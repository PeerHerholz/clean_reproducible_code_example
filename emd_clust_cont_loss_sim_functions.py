

# Import necessary modules
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.mixture import BayesianGaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from joblib import dump
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

np.random.seed(42)
torch.manual_seed(42)


# Define utility function to create paths
def create_output_paths(paths):
    for path in paths:
        if not os.path.isdir(path):
            os.makedirs(path)
    return paths


# Define utility function to simulate data
def simulate_data(
    seed=42, size=(2000, 100), save_path="./outputs/data/raw_data_sim.npy"
):
    np.random.seed(seed)
    data = np.random.randn(*size)
    np.save(save_path, data)
    return data


# Define utility function to data plots
def plot_data_samples(data, save_path="./outputs/graphics/data_examples.png"):
    fig, axes = plt.subplots(5, 1, sharex=True, figsize=(16, 10))
    for sample, ax in zip(np.arange(5), axes.flat):
        sns.heatmap(
            data[sample].reshape(-1, 100),
            cbar=False,
            annot=False,
            xticklabels=False,
            yticklabels=False,
            ax=ax,
        )
        ax.set_title(f"data sample {sample+1}")
    fig.savefig(save_path)


# Define class for the Encoder
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim):
        super(Encoder, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, x):
        return self.layers(x)


# Define class for the MLP
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        return self.layers(x)


# Define class for the contrastive loss
class ContrastiveLoss(nn.Module):
    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = (output1 - output2).pow(2).sum(1)
        return torch.mean(
            (1 - label) * euclidean_distance
            + label * torch.clamp(self.margin - euclidean_distance, min=0.0)
        )


# Define utility function to train the encoder
def train_encoder(encoder, data_train, epochs=4000, lr=0.001, momentum=0.9):
    loss_function = nn.MSELoss()
    optimizer = optim.SGD(encoder.parameters(), lr=lr, momentum=momentum)
    losses = []
    representations_during_training = []

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = encoder(data_train)
        out_features = encoder.layers[-1].out_features
        loss = loss_function(outputs, data_train[:, :out_features])
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        # Save representations every 1000 epochs
        if epoch % 1000 == 0:
            with torch.no_grad():
                representations_during_training.append(
                    encoder(data_train).cpu().numpy()
                )

    return losses, representations_during_training


# Define utility function to plot loss
def plot_loss(losses, title, save_path):
    fig, ax = plt.subplots()
    sns.lineplot(x=range(len(losses)), y=losses, ax=ax)
    sns.despine(offset=10, ax=ax)
    plt.title(title)
    plt.xlabel("Epoch number")
    plt.ylabel("Training loss")
    fig.savefig(save_path)


# Define utility function to plot encoder representations
def plot_encoder_representations(representations, save_path):
    fig, axes = plt.subplots(1, 5, sharex=True, figsize=(10, 2))
    for sample, ax in zip(np.arange(5), axes.flat):
        sns.heatmap(
            representations[sample].reshape(-1, 5),
            cbar=False,
            annot=False,
            xticklabels=False,
            yticklabels=False,
            ax=ax,
        )
        ax.set_title(f"Sample {sample+1}")
    fig.savefig(save_path)


# Define utility function to apply clustering
def apply_clustering(data, save_path="./outputs/models/dpgmm.joblib",
                     n_components=10):
    dpgmm = BayesianGaussianMixture(
        n_components=n_components,
        covariance_type="full",
        weight_concentration_prior_type="dirichlet_process",
        weight_concentration_prior=0.1,
        random_state=42,
    )
    dpgmm.fit(data)
    dump(dpgmm, save_path)
    return dpgmm


# Define utility function to plot TSNE
def plot_tsne(data, labels, save_path):
    # Calculate appropriate perplexity based on data size
    n_samples = len(data)
    perplexity = min(40, max(5, (n_samples - 1) // 3))
    
    tsne = TSNE(n_components=2, verbose=1, perplexity=perplexity,
                n_iter=300, random_state=42)
    tsne_results = tsne.fit_transform(data)

    df = pd.DataFrame()
    df["t-SNE dim 1"] = tsne_results[:, 0]
    df["t-SNE dim 2"] = tsne_results[:, 1]
    df["cluster"] = labels

    plt.figure(figsize=(16, 10))
    sns.scatterplot(
        x="t-SNE dim 1",
        y="t-SNE dim 2",
        palette=sns.color_palette("hls", len(set(labels))),
        data=df,
        hue="cluster",
        legend="full",
        alpha=0.3,
    )
    sns.despine(offset=10)
    plt.savefig(save_path)
    plt.close()


# Define utility function to generate positive and negative pairs
def generate_pairs(representations, cluster_assignments):
    clusters = []
    for i in range(cluster_assignments.max() + 1):
        clusters.append(representations[cluster_assignments == i])

    positive_pairs = []
    negative_pairs = []

    for i, cluster in enumerate(clusters):
        for j in range(len(cluster)):
            for k in range(j + 1, len(cluster)):
                positive_pairs.append((cluster[j], cluster[k]))
        for j in range(len(cluster)):
            for k in range(i + 1, len(clusters)):
                for m in range(len(clusters[k])):
                    negative_pairs.append((cluster[j], clusters[k][m]))

    positive_pairs_torch = torch.from_numpy(np.array(positive_pairs)).float()
    negative_pairs_torch = torch.from_numpy(np.array(negative_pairs)).float()

    return positive_pairs_torch, negative_pairs_torch


# Define utility function to train contrastive model
def train_contrastive_model(
    contrastive_model, positive_pairs, negative_pairs, epochs=200, lr=0.001
):
    optimizer = optim.Adam(contrastive_model.parameters(), lr=lr)
    contrastive_loss = ContrastiveLoss(margin=1.0)
    losses = []

    for epoch in range(epochs):
        optimizer.zero_grad()
        positive_pair_rep_1 = contrastive_model(positive_pairs[:, 0])
        positive_pair_rep_2 = contrastive_model(positive_pairs[:, 1])
        negative_pair_rep_1 = contrastive_model(negative_pairs[:, 0])
        negative_pair_rep_2 = contrastive_model(negative_pairs[:, 1])

        loss_positive = contrastive_loss(positive_pair_rep_1,
                                         positive_pair_rep_2, 0)
        loss_negative = contrastive_loss(negative_pair_rep_1,
                                         negative_pair_rep_2, 1)
        loss = loss_positive + loss_negative

        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return losses


# Define analysis steps
def main():
    # Set up output directories
    paths = ["./outputs/", "./outputs/data",
             "./outputs/graphics", "./outputs/models"]
    create_output_paths(paths)

    # Simulate data
    data = simulate_data()

    # Plot data samples
    plot_data_samples(data)

    # Train Encoder
    data_train, data_test = train_test_split(data, test_size=0.2,
                                             random_state=42)
    data_train_torch = torch.from_numpy(data_train).float()
    encoder = Encoder(input_dim=100, hidden_dim=100, embedding_dim=50)
    losses, representations_during_training = train_encoder(encoder,
                                                            data_train_torch)
    torch.save(encoder, "./outputs/models/encoder.pth")
    plot_loss(losses, "Loss of Encoder",
              "./outputs/graphics/loss_training.png")

    # Plot encoder representations at epoch 3000
    plot_encoder_representations(
        representations_during_training[3],
        "./outputs/graphics/data_representations_examples.png",
    )

    # Apply clustering
    dpgmm = apply_clustering(representations_during_training[3])
    cluster_assignments_train = dpgmm.predict(
        representations_during_training[3]
    )

    # Generate positive and negative pairs from clusters
    positive_pairs_torch, negative_pairs_torch = generate_pairs(
        representations_during_training[3], cluster_assignments_train
    )

    # Train contrastive model
    contrastive_model = MLP(input_dim=50, hidden_dim=50)
    losses_contrastive = train_contrastive_model(
        contrastive_model, positive_pairs_torch, negative_pairs_torch
    )
    torch.save(contrastive_model, "./outputs/models/contrastive_model.pth")
    plot_loss(
        losses_contrastive,
        "Loss of MLP with contrastive learning",
        "./outputs/graphics/loss_training_MLP_cont_learn.png",
    )

    # t-SNE visualization of cluster assignments in training data
    plot_tsne(
        representations_during_training[3],
        cluster_assignments_train,
        "./outputs/graphics/tsne_rep_clust_train.png",
    )

    # Apply trained encoder to test data
    data_test_torch = torch.from_numpy(data_test).float()
    encoder_embeddings_test = encoder(data_test_torch).detach().numpy()

    # Predict clusters for test data
    cluster_assignments_test = dpgmm.predict(encoder_embeddings_test)

    # Apply contrastive model to test data
    contrastive_representations_test = (
        contrastive_model(torch.from_numpy(encoder_embeddings_test).float())
        .detach()
        .numpy()
    )

    # t-SNE visualization of test data clusters
    plot_tsne(
        contrastive_representations_test,
        cluster_assignments_test,
        "./outputs/graphics/tsne_rep_clust_test.png",
    )


if __name__ == "__main__":
    main()
