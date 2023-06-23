import numpy as np
import csv
import matplotlib.pyplot as plt

# Load data from files
file_list = ['animals.csv', 'countries.csv', 'fruits.csv', 'veggies.csv']

def load_data(file_list):
    data = []
    for filename in file_list:
        with open(filename, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)  # Remove the header row
            for row in reader:
                try:
                    row = [float(x) for x in row]  # Convert all values to floats
                    data.append(row)
                except ValueError:
                    print(f"Skipping row {row}: could not convert to float")
    return data


def som(data, k, num_epochs):
    # Initialize grid of neurons
    neurons = np.random.rand(k, data.shape[1])

    # Iterate over data for specified number of epochs
    for epoch in range(num_epochs):
        # Shuffle data
        np.random.shuffle(data)

        # Iterate over each data point
        for i, x in enumerate(data):
            # Find best matching unit
            distances = np.linalg.norm(x - neurons, axis=1)
            bmu_index = np.argmin(distances)

            # Update weights of BMU and its neighbors
            for j in range(k):
                distance_to_bmu = np.abs(j - bmu_index)
                if distance_to_bmu <= 1:
                    learning_rate = 1.0 - epoch/num_epochs
                    neurons[j,:] += learning_rate * (x - neurons[j,:])

    # Assign each data point to its nearest neuron
    cluster_labels = np.zeros(data.shape[0])
    for i, x in enumerate(data):
        distances = np.linalg.norm(x - neurons, axis=1)
        cluster_labels[i] = np.argmin(distances)

    return cluster_labels

# Define function to compute purity
def purity(y_true, y_pred):
    contingency_matrix = np.zeros((np.max(y_true)+1, np.max(y_pred)+1))
    for i in range(len(y_true)):
        contingency_matrix[y_true[i], y_pred[i]] += 1
    return np.sum(np.max(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

# Define range of k values to test
k_values = range(2, 10)

# Initialize arrays to store precision, recall, and F-score for each k
precision = np.zeros(len(k_values))
recall = np.zeros(len(k_values))
fscore = np.zeros(len(k_values))


def evaluate_som(data, k_values, n_iterations):
    # Initialize lists to store the evaluation metrics
    purities = []
    avg_precision = []
    recalls = []
    f_scores = []

    for k in k_values:
        # Run the SOM algorithm to obtain k clusters
        neurons, bmu_indices = som(data, k, n_iterations)

        # Assign each data instance to the cluster corresponding to the nearest neuron
        clusters = [[] for _ in range(k)]
        for i, bmu_index in enumerate(bmu_indices):
            clusters[bmu_index].append(i)

        # Compute the purity of the clustering
        cluster_sizes = [len(c) for c in clusters]
        majority_classes = [np.argmax(np.bincount([data[i].label for i in c])) for c in clusters]
        purity = np.sum(np.multiply(cluster_sizes, majority_classes)) / len(data)
        purities.append(purity)



def evaluate_clusters(clusters, labels):
    # Compute the number of true positives, false positives, true negatives, and false negatives for each cluster
    cluster_metrics = []
    for c in clusters:
        tp = fp = tn = fn = 0
        for i in range(len(labels)):
            for j in range(i+1, len(labels)):
                if i in c and j in c and labels[i] == labels[j]:
                    tp += 1
                elif i in c and j not in c and labels[i] == labels[j]:
                    fp += 1
                elif i not in c and j not in c and labels[i] != labels[j]:
                    tn += 1
                elif i not in c and j in c and labels[i] == labels[j]:
                    fn += 1
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
        cluster_metrics.append((precision, recall, f_score))

    # Compute the average precision, recall, and F-score for all clusters
    avg_precision = np.mean([m[0] for m in cluster_metrics])
    avg_recall = np.mean([m[1] for m in cluster_metrics])
    avg_f_score = np.mean([m[2] for m in cluster_metrics])

    return cluster_metrics, avg_precision, avg_recall, avg_f_score

# Plot results


plt.plot(k_values, precision , label='Precision')
plt.plot(k_values, recall, label='Recall')
plt.plot(k_values, fscore, label='F-score')
plt.legend()
plt.xlabel('Number of clusters (k)')
plt.ylabel('Score')
plt.show()