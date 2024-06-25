import matplotlib.pyplot as plt
import numpy as np
import time

# Define the vertices of the triangles
A = np.array([0, 0])
B = np.array([0, 1])
C = np.array([1, 0])
D = np.array([1, 1])

# Generate data points
def generate_datapoints(num_datapoints):
    # Class 0 points within rectangle ABC
    datapoints_class_0 = []
    while len(datapoints_class_0) < num_datapoints:
        s, t = np.random.rand(2)
        if s + t < 1:
            datapoints_class_0.append(A + s * (B - A) + t * (C - A))

    # Class 1 points within rectangle BCD
    datapoints_class_1 = []
    while len(datapoints_class_1) < num_datapoints:
        s, t = np.random.rand(2)
        if s + t < 1:
            datapoints_class_1.append(D - s * (D - C) - t * (D - B))

    # Convert lists to numpy arrays
    datapoints_class_0 = np.array(datapoints_class_0)
    datapoints_class_1 = np.array(datapoints_class_1)

    return datapoints_class_0, datapoints_class_1

# Sequential Minimum Optimization solver
class SMO_model:
    def __init__(self, tolerance=1e-5, C=0.01):
        self.alpha = None
        self.beta = 0.0
        self.tol = tolerance
        self.C = C
        self.X_train = None
        self.y_train = None

    def train(self, X, y, max_passes=1000):
        # Initialization
        self.alpha = np.zeros(X.shape[0])
        self.beta = 0.0
        self.X_train = X
        self.y_train = y
        loss_history = []

        # Start training
        passes = 0
        while (passes < max_passes):
            num_changed_alphas = 0
            for i in range(X.shape[0]):
                error_i = self.predict_each(X[i]) - y[i]
                if (y[i] * error_i < -self.tol and self.alpha[i] < self.C) or (y[i] * error_i > self.tol and self.alpha[i] > 0):
                    j = np.random.choice([k for k in range(X.shape[0]) if k != i])
                    error_j = self.predict_each(X[j]) - y[j]

                    alpha_i_old = self.alpha[i]
                    alpha_j_old = self.alpha[j]

                    if y[i] != y[j]:
                        L = max(0, self.alpha[j] - self.alpha[i])
                        H = min(1, self.alpha[j] - self.alpha[i] + 1)
                    else:
                        L = max(0, self.alpha[j] + self.alpha[i] - 1)
                        H = min(1, self.alpha[j] + self.alpha[i])

                    if L == H:
                        continue

                    eta = 2 * np.dot(X[i], X[j]) - np.dot(X[i], X[i]) - np.dot(X[j], X[j])
                    if eta >= 0:
                        continue

                    self.alpha[j] -= y[j] * (error_i - error_j) / eta
                    self.alpha[j] = np.clip(self.alpha[j], L, H)

                    if abs(self.alpha[j] - alpha_j_old) < 1e-5:
                        continue

                    self.alpha[i] += y[i] * y[j] * (alpha_j_old - self.alpha[j])

                    b1 = self.beta - error_i - y[i] * (self.alpha[i] - alpha_i_old) * np.dot(X[i], X[i]) - y[j] * (self.alpha[j] - alpha_j_old) * np.dot(X[i], X[j])
                    b2 = self.beta - error_j - y[i] * (self.alpha[i] - alpha_i_old) * np.dot(X[i], X[j]) - y[j] * (self.alpha[j] - alpha_j_old) * np.dot(X[i], X[j])

                    if 0 < self.alpha[i] < 1:
                        self.beta = b1
                    elif 0 < self.alpha[j] < 1:
                        self.beta = b2
                    else:
                        self.beat = (b1 + b2) / 2

                    num_changed_alphas += 1

            # Record Loss
            loss = self.calculate_loss()
            loss_history.append(loss)

            if num_changed_alphas == 0:
                passes = passes + 1
            else:
                passes = 0
        
        return loss_history

    def calculate_loss(self):
        N = self.X_train.shape[0]
        loss = (1/2) * np.sum([self.alpha[i] * self.alpha[j] * self.y_train[i] * self.y_train[j] * np.dot(self.X_train[i], self.X_train[j].T) for j in range(N) for i in range(N)]) - np.sum(self.alpha)
        #w = np.sum([self.alpha[i] * self.y_train[i] * self.X_train[i] for i in range(N)])
        #loss = 0.5 * np.dot(w.T, w)
        return loss

    def predict_each(self, x):
        return np.sum(self.alpha * self.y_train * np.dot(self.X_train, x)) + self.beta

    def predict(self, X):
        y_pred = np.zeros(X.shape[0])
        for j in range(X.shape[0]):
            y_pred[j] = np.sign(self.predict_each(X[j])).astype(int)
        return y_pred

    def coefficient(self):
        return self.alpha

def calculate_accuracy(prediction, ground_truth):
    return np.mean(prediction == ground_truth)

num_datapoints_list = [20, 50, 100]

for num_datapoints in num_datapoints_list:
    print(f'--- Number of data points in each class = {num_datapoints} ---')
    datapoints_class_0, datapoints_class_1 = generate_datapoints(num_datapoints)

    # Plot the data points
    plt.scatter(datapoints_class_0[:, 0], datapoints_class_0[:, 1], c='blue', label='Class 0')
    plt.scatter(datapoints_class_1[:, 0], datapoints_class_1[:, 1], c='red', label='Class 1')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.plot([B[0], C[0]], [B[1], C[1]], c='black', linestyle='--', label='Line BC')
    plt.title(f'Linearly Separable Dataset (Number of data points = {num_datapoints})')
    plt.legend()
    plt.show()

    # Generate the dataset
    X = np.concatenate((np.array(datapoints_class_0), np.array(datapoints_class_1)))
    y = np.concatenate((np.full(num_datapoints, -1), np.full(num_datapoints, 1)))

    # Create and train the classifier
    model = SMO_model(tolerance=1e-5, C=1.00)
    start_time = time.time()
    loss_history = model.train(X, y)
    end_time = time.time()
    training_time = end_time - start_time
    prediction = model.predict(X)

    # Print training time and accuracy
    print("Training Time:", training_time)
    print("Training Accuracy:", calculate_accuracy(prediction, y))

    # Plotting the training loss curve
    plt.plot(loss_history)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.show()

    # Generate a meshgrid for plotting the classifier
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 20), np.linspace(y_min, y_max, 20))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot the classifier and data points
    plt.contourf(xx, yy, Z, alpha=0.5, cmap='rainbow')

    # Plot the data points
    plt.scatter(datapoints_class_0[:, 0], datapoints_class_0[:, 1], c='blue', label='Class 0')
    plt.scatter(datapoints_class_1[:, 0], datapoints_class_1[:, 1], c='red', label='Class 1')

    # Plot the misclassified data points
    misclassified_points = X[prediction != y]
    plt.scatter(misclassified_points[:, 0], misclassified_points[:, 1], c='black', marker='x', label='Misclassified')

    # Mark the support vectors
    support_vectors = model.X_train[model.alpha > 0]
    plt.scatter(support_vectors[:, 0], support_vectors[:, 1], s=100, facecolors='none', edgecolors='black', label='Support Vectors')

    # Show the plot
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.plot([B[0], C[0]], [B[1], C[1]], c='black', linestyle='--', label='Line BC')
    plt.title(f'Linearly Separable Dataset with Classifier (Number of data points = {num_datapoints})')
    plt.legend()
    plt.show()