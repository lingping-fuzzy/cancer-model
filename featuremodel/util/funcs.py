import torch
from sklearn.decomposition import PCA

from .optht import optht
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from sklearn.cluster import  KMeans
from collections import Counter
import numpy as np

def get_sample(method= 'random', feature_reduced=  None, n_clusters= 500, indices_4=None):
    if method == 'random':
        selected_indices = indices_4[torch.randperm(len(indices_4))[:500]]
    elif method == 'cluster':
        # kmeans select
        kmeans = KMeans(n_clusters= n_clusters, random_state=0)
        kmeans.fit(feature_reduced)
        representative_train=[]

        for i in range(n_clusters):
            cluster_indices = np.where(kmeans.labels_ == i)[0]
            # Select one representative image per cluster (e.g., the closest to the cluster center)
            closest_index = cluster_indices[np.argmin(np.linalg.norm( feature_reduced[cluster_indices] - kmeans.cluster_centers_[i], axis=1))]
            representative_train.append(closest_index)
        selected_indices =  indices_4[representative_train]
    elif method == 'clusterpca':
        # Optional: Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(feature_reduced)
        u, s, vh = np.linalg.svd(features_scaled, full_matrices=False)

        k = optht(features_scaled, sv=s, sigma=None)

        # Apply PCA for dimensionality reduction (optional)
        pca = PCA(n_components= k)
        features_ = pca.fit_transform(features_scaled)

        kmeans = KMeans(n_clusters= n_clusters, random_state=0)
        kmeans.fit(features_)
        representative_train=[]

        for i in range(n_clusters):
            cluster_indices = np.where(kmeans.labels_ == i)[0]
            # Select one representative image per cluster (e.g., the closest to the cluster center)
            closest_index = cluster_indices[np.argmin(np.linalg.norm( features_[cluster_indices] - kmeans.cluster_centers_[i], axis=1))]
            representative_train.append(closest_index)
        selected_indices =  indices_4[representative_train]
    elif method =='clusterconvex':
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(feature_reduced)

        # Apply PCA for dimensionality reduction (optional)
        pca = PCA(n_components= int(n_clusters/8))
        features_ = pca.fit_transform(features_scaled)

        kmeans = KMeans(n_clusters= n_clusters, random_state=0)
        kmeans.fit(features_)
        representative_train=[]

        for i in range(n_clusters):
            cluster_indices = np.where(kmeans.labels_ == i)[0]
            # Select one representative image per cluster (e.g., the closest to the cluster center)
            closest_index = cluster_indices[np.argmin(np.linalg.norm( features_[cluster_indices] - kmeans.cluster_centers_[i], axis=1))]
            representative_train.append(closest_index)
        selected_indices =  indices_4[representative_train]

    else:
        print('wrong sampling method')
    return selected_indices


def center_data(filtered_train_features, train_features_val):
    # Convert to numpy for easier manipulation with sklearn and scipy
    train_features_np = filtered_train_features.numpy()
    val_features_np = train_features_val.numpy()

    # Centering process: calculate mean and std, and then center the train data
    scaler = StandardScaler()
    train_features_centered = scaler.fit_transform(train_features_np)
    val_features_centered = scaler.transform(val_features_np)
    return train_features_centered, val_features_centered, scaler




def find_closest_group_voting(proj_val_point, grouped_proj_train, k=5):
    distances_labels = []

    for label, group in grouped_proj_train.items():
        # Calculate the distance between proj_val_point and all points in the group
        distances = cdist([proj_val_point], group, metric='euclidean')
        # Collect distances and corresponding labels
        for distance in distances[0]:
            distances_labels.append((distance, label))

    # Sort by distance and select the closest 5
    distances_labels.sort(key=lambda x: x[0])
    closest_5 = distances_labels[:k]

    # Extract the labels of the closest 5 points
    closest_labels = [label for _, label in closest_5]

    # Determine the majority label
    majority_label = Counter(closest_labels).most_common(1)[0][0]

    return majority_label


# function return low-D train features and Transform projection matrix
def get_SVD(train_features_centered, filtered_train_labels):
    # Memorize the mean and std
    # Perform SVD
    u, s, vh = np.linalg.svd(train_features_centered, full_matrices=False)
    v = vh.T

    k = optht(train_features_centered, sv=s, sigma=None)
    # Project train and validation data by V
    proj_train = np.dot(train_features_centered, v[:, :k])

    # Group proj_train based on train_labels
    grouped_proj_train = {}
    unique_labels = np.unique(filtered_train_labels.numpy())

    for label in unique_labels:
        mask = (filtered_train_labels.numpy() == label)
        grouped_proj_train[label] = proj_train[mask]
    return grouped_proj_train, v[:, :k]


def _fit_transform(proj_, val_features_centered):
    proj_val = np.dot(val_features_centered, proj_)
    return proj_val


def classify_voting(grouped_proj_train, proj_val, k=5):
    # Verify which group each data point in proj_val is closest to
    closest_groups = [find_closest_group_voting(val_point, grouped_proj_train, k=k) for val_point in proj_val]

    # Print the closest group labels for the validation data
    #     print(closest_groups)
    return closest_groups


def classify_probs(grouped_proj_train, proj_val, unique_labels, k=7):
    # Verify which group each data point in proj_val is closest to
    classification_probabilities = [find_closest_group_probs(val_point, grouped_proj_train, unique_labels, k=k) for
                                    val_point in proj_val]

    # Print the closest group labels for the validation data

    return classification_probabilities

def classify_(grouped_proj_train, proj_val):

    # Verify which group each data point in proj_val is closest to
    closest_groups = [find_closest_group(val_point, grouped_proj_train) for val_point in proj_val]

    # Print the closest group labels for the validation data
    #     print(closest_groups)
    return closest_groups


# Function to find the closest group in proj_train for a given proj_val point
def find_closest_group(proj_val_point, grouped_proj_train):
    min_distance = float('inf')
    closest_label = None
    for label, group in grouped_proj_train.items():
        # Calculate the distance between proj_val_point and all points in the group
        distances = cdist([proj_val_point], group, metric='euclidean')
        # Find the minimum distance to the group
        min_group_distance = distances.min()
        if min_group_distance < min_distance:
            min_distance = min_group_distance
            closest_label = label
    return closest_label

def find_closest_group_probs(proj_val_point, grouped_proj_train, unique_labels, k=7):
    distances_labels = []

    for label, group in grouped_proj_train.items():
        # Calculate the distance between proj_val_point and all points in the group
        distances = cdist([proj_val_point], group, metric='euclidean')
        # Collect distances and corresponding labels
        for distance in distances[0]:
            distances_labels.append((distance, label))

    # Sort by distance and select the closest 5
    distances_labels.sort(key=lambda x: x[0])
    closest_5 = distances_labels[:k]

    # Extract the labels and distances of the closest 5 points
    closest_labels = [label for _, label in closest_5]
    closest_distances = [distance for distance, _ in closest_5]

    # Count the number of occurrences of each label in the closest 5
    count_labels = Counter(closest_labels)

    # Prepare the output arrays
    label_count_array = np.zeros(len(unique_labels))
    average_distance_array = np.zeros(len(unique_labels))

    for i, label in enumerate(unique_labels):
        label_count_array[i] = count_labels[label]
        if count_labels[label] > 0:
            label_distances = [distances_labels[j][0] for j in range(k) if closest_labels[j] == label]
            average_distance_array[i] = np.mean(label_distances)
        else:
            average_distance_array[i] = np.sum(closest_distances)

    #     return label_count_array, average_distance_array
    classification_probabilities = np.exp(-average_distance_array) / np.sum(np.exp(-average_distance_array))
    return classification_probabilities

# Example usage:
#
# label_counts, average_distances = find_closest_group(proj_val[0], grouped_proj_train, unique_labels)

