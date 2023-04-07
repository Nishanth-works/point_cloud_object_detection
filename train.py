import os
import numpy as np
import open3d as o3d
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, TensorBoard
from pointnet import PointNet
import datetime

# Define constants
NUM_CLASSES = 10
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001
ROTATION_RANGE = [0, 360]
SCALING_RANGE = [0.8, 1.2]
JITTER_RANGE = 0.02

# Define function to preprocess point cloud data
def preprocess_data(pcd):
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
def preprocess_labels(labels):
    # Convert labels to integers
    labels = labels.astype(int)
    # Convert labels to one-hot encoding
    encoder = OneHotEncoder(categories='auto')
    labels = encoder.fit_transform(labels.reshape(-1, 1)).toarray()
    return labels

# Define training generator
def training_generator(train_data, train_labels):
    while True:
        # Shuffle data and labels
        idx = np.random.permutation(len(train_data))
        train_data = train_data[idx]
        train_labels = train_labels[idx]
        # Generate batches of data
        for i in range(0, len(train_data), BATCH_SIZE):
            batch_data = np.zeros((BATCH_SIZE, 1024, 3))
            batch_labels = np.zeros((BATCH_SIZE, NUM_CLASSES))
            for j in range(BATCH_SIZE):
                # Load and preprocess data and labels
                pcd = o3d.io.read_point_cloud(train_data[i+j])
                labels = np.load(train_labels[i+j])
                batch_data[j] = preprocess_data(pcd)
                batch_labels[j] = preprocess_labels(labels)
            yield batch_data, batch_labels

# Define learning rate schedule
def lr_schedule(epoch, lr):
    if epoch < 50:
        return lr
    elif epoch < 75:
        return lr * 0.1
    else:
        return lr * 0.01

# Define checkpoint and early stopping callbacks
checkpoint = ModelCheckpoint('pointnet_weights.h5', save_best_only=True, save_weights_only=True, verbose=1)
early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

# Define TensorBoard callback
logdir = os.path.join('logs', 'fit', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = TensorBoard(log_dir=logdir, histogram_freq=1)

# Load training data and labels
train_data = np.array([os.path.join('data/train', f) for f in os.listdir('data/train') if f.endswith('.ply')])
train_labels = np.array([os.path.join('data/train', f) for f in os.listdir('data/train') if f.endswith('.npy')])

# Split data into training and validation sets
num_train = int(len(train_data) * 0.8)
train_data, val_data = train_data[:num_train], train_data[num_train:]
train_labels, val_labels = train_labels[:num_train], train_labels[num_train:]

# Create and compile model
model = PointNet(NUM_CLASSES)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss='categorical_crossentropy', metrics=['accuracy'])

# Define learning rate schedule callback
lr_schedule_callback = LearningRateScheduler(lr_schedule)

# Train model using generator and callbacks
model.fit(training_generator(train_data, train_labels),
          epochs=EPOCHS,
          steps_per_epoch=len(train_data)//BATCH_SIZE,
          validation_data=(preprocess_data(val_data), preprocess_labels(val_labels)),
          callbacks=[checkpoint, early_stop, tensorboard_callback, lr_schedule_callback])

# Load best weights from checkpoint
model.load_weights('pointnet_weights.h5')

# Evaluate model on test data
test_data = np.array([os.path.join('data/test', f) for f in os.listdir('data/test') if f.endswith('.ply')])
test_labels = np.array([os.path.join('data/test', f) for f in os.listdir('data/test') if f.endswith('.npy')])
test_loss, test_acc = model.evaluate(preprocess_data(test_data), preprocess_labels(test_labels), verbose=0)

# Print test loss and accuracy
print('Test loss:', test_loss)
print('Test accuracy:', test_acc)