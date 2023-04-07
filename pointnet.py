import tensorflow as tf

class PointNet(tf.keras.Model):
    def __init__(self, num_classes, dropout_rate=0.3):
        super(PointNet, self).__init__()
        self.num_classes = num_classes
        
        # Define MLP layers
        self.mlp1 = tf.keras.layers.Conv1D(filters=64, kernel_size=1, activation='relu')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.mlp2 = tf.keras.layers.Conv1D(filters=128, kernel_size=1, activation='relu')
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.mlp3 = tf.keras.layers.Conv1D(filters=1024, kernel_size=1, activation='relu')
        self.bn3 = tf.keras.layers.BatchNormalization()
        
        # Define max pooling layer
        self.max_pool = tf.keras.layers.GlobalMaxPooling1D()
        
        # Define additional fully connected layers
        self.fc1 = tf.keras.layers.Dense(512, activation='relu')
        self.bn4 = tf.keras.layers.BatchNormalization()
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.fc2 = tf.keras.layers.Dense(256, activation='relu')
        self.bn5 = tf.keras.layers.BatchNormalization()
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
        self.fc3 = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        # Apply data augmentation
        inputs = self.augment(inputs)
        
        # Pass inputs through MLP layers
        x = self.mlp1(inputs)
        x = self.bn1(x)
        x = self.mlp2(x)
        x = self.bn2(x)
        x = self.mlp3(x)
        x = self.bn3(x)
        
        # Apply max pooling
        x = self.max_pool(x)
        
        # Pass through additional fully connected layers
        x = self.fc1(x)
        x = self.bn4(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.bn5(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x
    
    def augment(self, inputs):
        # Apply random rotation
        theta = tf.random.uniform([1], minval=0, maxval=2*tf.math.pi)
        rotation_matrix = tf.stack([tf.cos(theta), -tf.sin(theta), tf.sin(theta), tf.cos(theta)])
        rotation_matrix = tf.reshape(rotation_matrix, [2, 2])
        inputs = tf.matmul(inputs, rotation_matrix)
        
        # Apply random scaling
        scale = tf.random.uniform([1], minval=0.8, maxval=1.2)
        inputs *= scale
        
        # Apply random translation
        translation = tf.random.uniform([3], minval=-0.02, maxval=0.02)
        inputs += translation
        
        return inputs