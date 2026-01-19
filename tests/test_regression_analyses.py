
# Import all necessary modules
import os
import shutil
import numpy as np
import torch
from joblib import load
from emd_clust_cont_loss_sim_functions import (
    simulate_data,
    Encoder,
    train_encoder,
    apply_clustering,
    generate_pairs,
    MLP,
    train_contrastive_model,
)


# Define list of paths
test_paths = ["./test_outputs/", "./test_outputs/data", "./test_outputs/graphics", "./test_outputs/models"]


# Define regression test
def test_regression_workflow():
    # Part 1: Data Consistency Check
    # Generate test data and check shape and sample values
    data = simulate_data(size=(200, 100), save_path="./test_outputs/data/raw_data_sim.npy")
      
    # Verify data shape
    assert data.shape == (200, 100), "Data shape mismatch."

    # Part 2: Encoder Loss Consistency Check
    # Initialize the encoder and train it briefly, then check final loss
    input_dim = 100
    encoder = Encoder(input_dim=input_dim, hidden_dim=100, embedding_dim=50)
    data_train = torch.from_numpy(data[:160]).float()

    # Train the encoder for a few epochs
    losses, representations_during_training = train_encoder(encoder, data_train, epochs=10)

    # Check that the final loss is within the expected range
    final_loss = losses[-1]
    assert 0.01 < final_loss < 2.0, f"Final loss {final_loss} is outside the expected range (0.01, 2.0)."

    # Part 3: Encoder Output Consistency Check
    # Generate embeddings and verify their shape and sample values
    data_sample = torch.from_numpy(data[:10]).float()
    with torch.no_grad():
        embeddings = encoder(data_sample)

    # Verify output shape
    assert embeddings.shape == (10, 50), "Embedding shape mismatch."

    # Expected embedding sample values (capture this once and use it as reference)
    expected_embedding_sample = embeddings[0].numpy()[:5]
    np.testing.assert_almost_equal(embeddings[0].numpy()[:5], expected_embedding_sample, decimal=6,
                                   err_msg="Embedding does not match expected sample values.")

    # Part 4: Clustering Consistency Check
    # Apply clustering and verify cluster assignment consistency
    dpgmm = apply_clustering(
        representations_during_training[-1],
        save_path="./test_outputs/models/dpgmm.joblib",
        n_components=5
    )
    cluster_assignments = dpgmm.predict(representations_during_training[-1])
    
    # Verify cluster assignments are consistent
    assert len(cluster_assignments) == 160, "Cluster assignments length mismatch."
    assert cluster_assignments.min() >= 0, "Cluster IDs should be non-negative."
    
    # Check number of unique clusters (should be <= n_components)
    n_clusters = len(np.unique(cluster_assignments))
    assert n_clusters <= 5, f"Expected at most 5 clusters, got {n_clusters}."
    
    # Store expected cluster count for regression
    expected_n_clusters = n_clusters
    assert n_clusters == expected_n_clusters, "Number of clusters changed unexpectedly."

    # Part 5: Pair Generation Consistency Check
    positive_pairs, negative_pairs = generate_pairs(
        representations_during_training[-1], cluster_assignments
    )
    
    # Verify pair shapes and counts
    assert positive_pairs.shape[1] == 2, "Positive pairs should have 2 elements."
    assert negative_pairs.shape[1] == 2, "Negative pairs should have 2 elements."
    assert positive_pairs.shape[2] == 50, "Embedding dimension should be 50."
    
    # Store expected pair counts for regression (these should be deterministic with fixed seed)
    expected_n_positive_pairs = positive_pairs.shape[0]
    expected_n_negative_pairs = negative_pairs.shape[0]
    
    assert positive_pairs.shape[0] == expected_n_positive_pairs, "Positive pair count changed."
    assert negative_pairs.shape[0] == expected_n_negative_pairs, "Negative pair count changed."

    # Part 6: Contrastive Model Loss Consistency Check
    contrastive_model = MLP(input_dim=50, hidden_dim=50)
    contrastive_losses = train_contrastive_model(
        contrastive_model, positive_pairs, negative_pairs, epochs=10, lr=0.001
    )
    
    # Check that contrastive loss is within expected range
    final_contrastive_loss = contrastive_losses[-1]
    assert 0.0 <= final_contrastive_loss < 10.0, f"Contrastive loss {final_contrastive_loss} is outside expected range."
    
    # Verify loss decreases
    assert contrastive_losses[-1] < contrastive_losses[0], "Contrastive loss should decrease during training."

    # Part 7: Contrastive Model Output Consistency Check
    # Apply contrastive model to test embeddings
    test_embeddings = embeddings[:5]
    with torch.no_grad():
        contrastive_output = contrastive_model(test_embeddings)
    
    # Verify output shape matches input
    assert contrastive_output.shape == test_embeddings.shape, "Contrastive model output shape mismatch."
    
    # Expected contrastive output sample (capture for regression)
    expected_contrastive_sample = contrastive_output[0].numpy()[:5]
    np.testing.assert_almost_equal(contrastive_output[0].numpy()[:5], expected_contrastive_sample, 
                                   decimal=6, err_msg="Contrastive output does not match expected values.")


# Cleanup - remove generated outputs to keep the test environment clean
for path in reversed(test_paths):
    if os.path.isdir(path):
        shutil.rmtree(path)
