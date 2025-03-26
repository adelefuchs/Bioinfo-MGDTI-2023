import pandas as pd
import numpy as np
import cupy as cp  # Import CuPy for GPU acceleration
from Bio import pairwise2
from cupy import asnumpy  # To convert CuPy array back to NumPy if needed

# Load the protein sequence data
protein_sequences_file = 'origin_dataset/protein_sequences.csv'
protein_sequences_df = pd.read_csv(protein_sequences_file, header=None)

# Function to compute sequence similarity using Smith-Waterman alignment
# We use localxx for local alignment
def sequence_similarity(seq1, seq2):
    alignments = pairwise2.align.localxx(seq1, seq2)  # Local alignment
    if not alignments:
        return 0.0
    return alignments[0][2] / max(len(seq1), len(seq2))  # Normalized score

# Convert sequences into a list of strings for easy access
protein_sequences = protein_sequences_df[0].tolist()

# Use CuPy to store the similarity matrix on the GPU
n_proteins = len(protein_sequences)
similarity_matrix_proteins = cp.zeros((n_proteins, n_proteins), dtype=cp.float32)

# Function to calculate similarity on the GPU for all pairs
def compute_similarity_on_gpu():
    # Use a CuPy array to store similarity results
    for i in range(n_proteins):
        seq1 = protein_sequences[i]
        for j in range(i, n_proteins):
            seq2 = protein_sequences[j]
            similarity = sequence_similarity(seq1, seq2)
            similarity_matrix_proteins[i, j] = similarity
            similarity_matrix_proteins[j, i] = similarity  # Symmetric matrix

# Compute the similarity matrix using the GPU
compute_similarity_on_gpu()

# Convert the CuPy matrix back to a Pandas DataFrame
similarity_matrix_proteins_df = pd.DataFrame(asnumpy(similarity_matrix_proteins))

# Save as a .txt file (space-separated)
similarity_matrix_proteins_df.to_csv('origin_dataset/Similarity_Matrix_Proteins.txt', sep=' ', header=False, index=False)

print("Protein similarity matrix saved to Similarity_Matrix_Proteins.txt")
