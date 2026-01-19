
import os
import shutil
import numpy as np
import torch
from joblib import load
from sklearn.model_selection import train_test_split
from emd_clust_cont_loss_sim_functions import (
    create_output_paths,
    simulate_data,
    plot_data_samples,
    Encoder,
    train_encoder,
    plot_loss,
    plot_encoder_representations,
    apply_clustering,
    generate_pairs,
    MLP,
    train_contrastive_model,
    plot_tsne,
)

    
def main_test():
    """Test version of main with reduced parameters for faster execution."""
    # Set up output directories
    paths = ["./outputs/", "./outputs/data",
             "./outputs/graphics", "./outputs/models"]
    create_output_paths(paths)

    # Simulate data with smaller size
    data = simulate_data(size=(200, 100), 
                        save_path="./outputs/data/raw_data_sim.npy")

    # Plot data samples
    plot_data_samples(data)

    # Train Encoder with fewer epochs
    data_train, data_test = train_test_split(data, test_size=0.2,
                                             random_state=42)
    data_train_torch = torch.from_numpy(data_train).float()
    encoder = Encoder(input_dim=100, hidden_dim=100, embedding_dim=50)
    losses, representations_during_training = train_encoder(encoder,
                                                            data_train_torch,
                                                            epochs=10)
    torch.save(encoder, "./outputs/models/encoder.pth")
    plot_loss(losses, "Loss of Encoder",
              "./outputs/graphics/loss_training.png")

    # Plot encoder representations at last epoch
    plot_encoder_representations(
        representations_during_training[-1],
        "./outputs/graphics/data_representations_examples.png",
    )

    # Apply clustering with fewer components
    dpgmm = apply_clustering(representations_during_training[-1],
                            n_components=5)
    cluster_assignments_train = dpgmm.predict(
        representations_during_training[-1]
    )

    # Generate positive and negative pairs from clusters
    positive_pairs_torch, negative_pairs_torch = generate_pairs(
        representations_during_training[-1], cluster_assignments_train
    )

    # Train contrastive model with fewer epochs
    contrastive_model = MLP(input_dim=50, hidden_dim=50)
    losses_contrastive = train_contrastive_model(
        contrastive_model, positive_pairs_torch, negative_pairs_torch,
        epochs=10
    )
    torch.save(contrastive_model, "./outputs/models/contrastive_model.pth")
    plot_loss(
        losses_contrastive,
        "Loss of MLP with contrastive learning",
        "./outputs/graphics/loss_training_MLP_cont_learn.png",
    )

    # t-SNE visualization of cluster assignments in training data
    plot_tsne(
        representations_during_training[-1],
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

def test_system_workflow():
    # Set up paths for verification
    output_paths = ["./outputs/", "./outputs/data", "./outputs/graphics", "./outputs/models"]
    data_file = "./outputs/data/raw_data_sim.npy"
    plot_file = "./outputs/graphics/data_examples.png"
    encoder_file = "./outputs/models/encoder.pth"
    clustering_file = "./outputs/models/dpgmm.joblib"
    contrastive_model_file = "./outputs/models/contrastive_model.pth"
    tsne_train_plot = "./outputs/graphics/tsne_rep_clust_train.png"
    tsne_test_plot = "./outputs/graphics/tsne_rep_clust_test.png"
    contrastive_loss_plot = "./outputs/graphics/loss_training_MLP_cont_learn.png"
    
    # Run the test version of main function (faster with reduced parameters)
    main_test()
    
    # Check that output directories are created
    for path in output_paths:
        assert os.path.isdir(path), f"Expected directory {path} to be created."

    # Verify data file existence and format
    assert os.path.isfile(data_file), "Expected data file not found."
    data = np.load(data_file)
    assert isinstance(data, np.ndarray), "Data file is not in the expected .npy format."
    
    # Verify the plot file is created
    assert os.path.isfile(plot_file), "Expected plot file not found."

    # Verify the encoder model file is created
    assert os.path.isfile(encoder_file), "Expected encoder model file not found."
    
    # Step 1: Load and check encoder model
    encoder = torch.load(encoder_file)
    assert isinstance(encoder, Encoder), "The loaded model is not an instance of the Encoder class."
    
    # Step 2: Evaluate encoder model with sample data
    data_sample = torch.from_numpy(data[:10]).float()
    with torch.no_grad():
        embeddings = encoder(data_sample)
    
    # Check the output embeddings shape
    expected_shape = (10, encoder.layers[-1].out_features)
    assert embeddings.shape == expected_shape, f"Expected embeddings shape {expected_shape}, but got {embeddings.shape}."

    # Step 3: Check if training reduces loss over epochs
    data_train = torch.from_numpy(data[:80]).float()
    encoder_test = Encoder(input_dim=100, hidden_dim=100, embedding_dim=50)
    losses, _ = train_encoder(encoder_test, data_train, epochs=10)
    
    initial_loss, final_loss = losses[0], losses[-1]
    assert final_loss < initial_loss, "Expected final loss to be lower than initial loss, indicating training progress."

    # Step 4: Verify clustering model was created and is valid
    assert os.path.isfile(clustering_file), "Expected clustering model file not found."
    dpgmm = load(clustering_file)
    
    test_embeddings = encoder(data_sample).detach().numpy()
    cluster_assignments = dpgmm.predict(test_embeddings)
    assert isinstance(cluster_assignments, np.ndarray), "Cluster assignments should be numpy array."
    assert len(cluster_assignments) == 10, "Expected 10 cluster assignments."
    assert cluster_assignments.min() >= 0, "Cluster IDs should be non-negative."

    # Step 5: Verify contrastive model was created and is valid
    assert os.path.isfile(contrastive_model_file), "Expected contrastive model file not found."
    contrastive_model = torch.load(contrastive_model_file)
    assert isinstance(contrastive_model, MLP), "The loaded model is not an instance of the MLP class."
    
    with torch.no_grad():
        contrastive_output = contrastive_model(torch.from_numpy(test_embeddings).float())
    assert contrastive_output.shape == test_embeddings.shape, "Contrastive model output shape mismatch."

    # Step 6: Verify all expected plots were created
    assert os.path.isfile(tsne_train_plot), "Expected t-SNE training plot not found."
    assert os.path.isfile(tsne_test_plot), "Expected t-SNE test plot not found."
    assert os.path.isfile(contrastive_loss_plot), "Expected contrastive learning loss plot not found."

    # Cleanup
    for path in reversed(output_paths):
        if os.path.isdir(path):
            shutil.rmtree(path)
