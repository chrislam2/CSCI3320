import numpy as np
import matplotlib.pyplot as plt

# Step 1: Initialize parameters
K = 2  # Number of clusters
D = 2  # Number of dimensions
N = 500  # Number of data points per cluster

# Cluster 1 parameters
mu1 = np.array([0, 0])
cov1 = np.array([[1.0, 0.6], [0.6, 1]])

# Cluster 2 parameters
mu2 = np.array([3, 0])
cov2 = np.array([[0.5, 0.4], [0.4, 0.5]])

# Generate data points
data1 = np.random.multivariate_normal(mu1, cov1, N)
data2 = np.random.multivariate_normal(mu2, cov2, N)

# Combine data points
data = np.concatenate((data1, data2))

# Plot the data points
plt.scatter(data1[:, 0], data1[:, 1], color='orange', label='Cluster 1')
plt.scatter(data2[:, 0], data2[:, 1], color='blue', label='Cluster 2')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('GMM Data')
plt.legend()
plt.show()

# Step 2: Expectation-Maximization (EM) algorithm
# Initialize parameters
pi = np.ones(K) / K  # Mixing coefficients
mu = np.random.randn(K, D)  # Means
cov = np.zeros((K, D, D))  # Covariance matrices
for k in range(K):
    cov[k] = np.eye(D)

# E step
def compute_posterior(data, pi, mu, cov):
    N = data.shape[0]
    K = pi.shape[0]
    posterior = np.zeros((N, K))
    
    for i in range(N):
        for j in range(K):
            diff = data[i] - mu[j]
            cov_inv = np.linalg.inv(cov[j])
            exponent = -0.5 * np.dot(np.dot(diff, cov_inv), diff.T)
            normalization = np.sqrt(np.linalg.det(2 * np.pi * cov[j]))
            posterior[i, j] = pi[j] * np.exp(exponent) / normalization
        posterior[i] /= np.sum(posterior[i])
    
    return posterior

# M step
def update_parameters(data, posterior):
    N = data.shape[0]
    K = posterior.shape[1]
    
    # Update mixing coefficients
    pi = np.sum(posterior, axis=0) / N
    
    # Update means
    mu = np.dot(posterior.T, data) / np.sum(posterior, axis=0)[:, np.newaxis]
    
    # Update covariance matrices
    cov = np.zeros((K, D, D))
    for j in range(K):
        diff = data - mu[j]
        cov[j] = np.dot(posterior[:, j] * diff.T, diff) / np.sum(posterior[:, j])
    
    return pi, mu, cov

# Run EM algorithm
max_iterations = 100
tolerance = 1e-6

for iteration in range(max_iterations):
    # E step
    posterior = compute_posterior(data, pi, mu, cov)
    
    # M step
    new_pi, new_mu, new_cov = update_parameters(data, posterior)
    
    # Check convergence
    if np.max(np.abs(new_pi - pi)) < tolerance and np.max(np.abs(new_mu - mu)) < tolerance and np.max(np.abs(new_cov - cov)) < tolerance:
        print(f'Converged after {iteration + 1} iterations.')
        break
    
    # Update parameters
    pi = new_pi
    mu = new_mu
    cov = new_cov

# Print estimated parameters
print('Estimated Parameters:')
print('---------------------')
for k in range(K):
    print(f'Cluster {k+1}:')
    print('Mean:', mu[k])
    print('Covariance Matrix:')
    print(cov[k])
    print()