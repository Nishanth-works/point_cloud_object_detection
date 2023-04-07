import tensorflow as tf
import numpy as np
import open3d as o3d
from sklearn.preprocessing import OneHotEncoder
from pointnet import PointNet

# Define constants
NUM_CLASSES = 10
ROTATION_RANGE = [0, 360]
SCALING_RANGE = [0.8, 1.2]
JITTER_RANGE = 0.02

# Define function to preprocess point cloud data
def preprocess_data(filename):
    # Load point cloud from file
    pcd = o3d.io.read_point_cloud(filename)
    # Apply rotation, scaling, and jittering
    pcd.rotate(np.random.uniform(*ROTATION_RANGE), np.random.uniform(*ROTATION_RANGE), np.random.uniform(*ROTATION_RANGE))
    pcd.scale(np.random.uniform(*SCALING_RANGE))
    pcd.translate(np.random.uniform(-JITTER_RANGE, JITTER_RANGE, size=(3,)))
    # Remove points outside a specified radius from the origin
    pcd = pcd.crop(o3d.geometry.AxisAlignedBoundingBox([-1,-1,-1], [1,1,1]))
    # Downsample point cloud using voxel grid
    downsampled_pcd, _ = pcd.voxel_down_sample_and_trace(0.05, 0.05)
    # Convert point cloud to numpy array
    points = np.asarray(downsampled_pcd.points)
    # Normalize point cloud to have zero mean and unit variance
    points = (points - np.mean(points, axis=0)) / np.std(points, axis=0)
    return points

# Define function to preprocess labels
def preprocess_labels(filename):
    # Load labels from file
    labels = np.load(filename)
    # Convert labels to integers
    labels = labels.astype(int)
    # Convert labels to one-hot encoding
    encoder = OneHotEncoder(categories='auto')
    labels = encoder.fit_transform(labels.reshape(-1, 1)).toarray()
    return labels

# Load test data and labels
test_data_files = [f'data/test/{i}.ply' for i in range(100)]
test_label_files = [f'data/test/{i}.npy' for i in range(100)]
test_data = np.array([preprocess_data(filename) for filename in test_data_files])
test_labels = np.array([preprocess_labels(filename) for filename in test_label_files])

# Create PointNet model
model = PointNet(num_classes=NUM_CLASSES)

# Load saved model weights
model.load_weights('pointnet_weights.h5')

# Define testing generator to load data and labels
def testing_generator(data, labels):
    while True:
        for i in range(len(data)):
            # Load and preprocess data and labels
            batch_data = data[i]
            batch_labels = labels[i]
            # Reshape data and labels for model input
            batch_data = np.expand_dims(batch_data, axis=0)
            batch_labels = np.expand_dims(batch_labels, axis=0)
            yield batch_data, batch_labels

# Evaluate model on test data
test_loss, test_acc = model.evaluate(test_data, test_labels, verbose=0)
print('Test loss:', test_loss)
print('Test accuracy:', test_acc)

# Generate predictions on test data
test_preds = model.predict(test_data)
test_preds = np.argmax(test_preds, axis=1)

# Calculate confusion matrix
conf_matrix = tf.math.confusion_matrix(np.argmax(test_labels, axis=1), test_preds)
print('Confusion matrix:\n', conf_matrix.numpy())

# Calculate precision, recall, and F1 score
true_positives = np.diag(conf_matrix)
false_positives = np.sum(conf_matrix, axis=0) - true_positives
false_negatives = np.sum(conf_matrix, axis=1) - true_positives
precision = np.mean(true_positives / (true_positives + false_positives + 1e-9))
recall = np.mean(true_positives / (true_positives + false_negatives + 1e-9))
f1_score = 2 * precision * recall / (precision + recall + 1e-9)

print('Precision:', precision)
print('Recall:', recall)
print('F1 score:', f1_score)