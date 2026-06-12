import numpy as np
import matplotlib.pyplot as plt

# --------------------------
# Load Dataset (numpy only)
# --------------------------

# Columns: ID, Year, Country, Continent, Population, GDP, Life_Expectancy,
# Urban_Population_Percent, Internet_Users_Percent
# We use only the numeric columns and drop ID (just an identifier)
raw = np.genfromtxt(
    "Data_Africa.csv",
    delimiter=",",
    skip_header=1,
    usecols=(1, 4, 5, 6, 7, 8)
)

print("Original Data Shape:")
print(raw.shape)

print("\nMissing Values per Column:")
print(np.isnan(raw).sum(axis=0))

# --------------------------
# Handle Missing Values (fill with column mean)
# --------------------------

col_means = np.nanmean(raw, axis=0)
inds = np.where(np.isnan(raw))
raw[inds] = np.take(col_means, inds[1])

print("\nMissing Values After Cleaning:")
print(np.isnan(raw).sum(axis=0))

X = raw

# --------------------------
# Standardize Data
# --------------------------

mean = np.mean(X, axis=0)
std = np.std(X, axis=0)

X_standardized = (X - mean) / std

print("\nStandardized Data Shape:")
print(X_standardized.shape)

# --------------------------
# Visualize Data Before PCA
# --------------------------

plt.figure(figsize=(8, 6))
plt.scatter(X_standardized[:, 0], X_standardized[:, 1])
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Data Before PCA")
plt.grid(True)
plt.show()

# --------------------------
# Covariance Matrix
# --------------------------

cov_matrix = np.cov(X_standardized, rowvar=False)

print("\nCovariance Matrix:")
print(cov_matrix)

# --------------------------
# Eigenvalues & Eigenvectors
# (eigh used because the covariance matrix is symmetric)
# --------------------------

eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

print("\nEigenvalues:")
print(eigenvalues)

print("\nEigenvectors:")
print(eigenvectors)

# --------------------------
# Sort Eigenvalues (descending)
# --------------------------

sorted_indices = np.argsort(eigenvalues)[::-1]
sorted_eigenvalues = eigenvalues[sorted_indices]
sorted_eigenvectors = eigenvectors[:, sorted_indices]

print("\nSorted Eigenvalues:")
print(sorted_eigenvalues)

# --------------------------
# Explained Variance
# --------------------------

explained_variance_ratio = sorted_eigenvalues / np.sum(sorted_eigenvalues)

print("\nExplained Variance Ratio:")
print(explained_variance_ratio)

# --------------------------
# Cumulative Variance
# --------------------------

cumulative_variance = np.cumsum(explained_variance_ratio)

print("\nCumulative Variance:")
print(cumulative_variance)

# --------------------------
# Dynamic Component Selection (95% threshold)
# --------------------------

threshold = 0.95
num_components = np.argmax(cumulative_variance >= threshold) + 1

print(f"\nNumber of Components Selected: {num_components}")

# --------------------------
# PCA Projection
# --------------------------

principal_components = sorted_eigenvectors[:, :num_components]
reduced_data = np.dot(X_standardized, principal_components)

print("\nReduced Data Shape:")
print(reduced_data.shape)

print("\nReduced Data Preview:")
print(reduced_data[:5])

print("\nPrincipal Components:")
print(principal_components)

# --------------------------
# PCA Visualization
# --------------------------

plt.figure(figsize=(8, 6))
plt.scatter(reduced_data[:, 0], reduced_data[:, 1])
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Data After PCA")
plt.grid(True)
plt.show()

# --------------------------
# Scree Plot
# --------------------------

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(sorted_eigenvalues) + 1), explained_variance_ratio, marker="o")
plt.xlabel("Principal Component")
plt.ylabel("Explained Variance Ratio")
plt.title("Scree Plot")
plt.grid(True)
plt.show()

# --------------------------
# Cumulative Variance Plot
# --------------------------

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker="o")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("Cumulative Variance Explained")
plt.grid(True)
plt.show()

# --------------------------
# Final Results
# --------------------------

print("\n========== PCA RESULTS ==========")
print("\nNumber of Components:")
print(num_components)
print("\nSorted Eigenvalues:")
print(sorted_eigenvalues)
print("\nExplained Variance Ratio:")
print(explained_variance_ratio)
print("\nCumulative Variance:")
print(cumulative_variance)