"""
Hybrid Soil Classification Model
Combines CNN (image features) and LSTM (temporal features) for enhanced prediction
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import (Dense, Dropout, LSTM, Input, 
                                    Concatenate, BatchNormalization)
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
import pickle
import os


class HybridSoilClassifier:
    """
    Hybrid model combining:
    - CNN: Image-based soil type classification
    - LSTM: Temporal soil characteristics (moisture, pH, temperature, etc.)
    """
    
    def __init__(self, cnn_model_path=None, lstm_weights_path=None):
        self.cnn_model_path = cnn_model_path or "models/soil_classifier_93_86.h5"
        self.lstm_weights_path = lstm_weights_path or "models/lstm_soil_model.h5"
        self.scaler_path = "models/soil_feature_scaler.pkl"
        
        self.num_classes = 4  # Alluvial, Black, Clay, Red
        self.temporal_features = 7  # Number of temporal features
        self.sequence_length = 5  # Number of timesteps for LSTM
        
        self.cnn_model = None
        self.hybrid_model = None
        self.scaler = None
        
    def load_cnn_model(self):
        """Load pre-trained CNN model"""
        from tensorflow.keras.layers import DepthwiseConv2D
        
        class CustomDepthwiseConv2D(DepthwiseConv2D):
            def __init__(self, **kwargs):
                kwargs.pop('groups', None)
                super().__init__(**kwargs)
        
        custom_objects = {'DepthwiseConv2D': CustomDepthwiseConv2D}
        self.cnn_model = load_model(self.cnn_model_path, 
                                     custom_objects=custom_objects, 
                                     compile=False)
        
        # Create a feature extraction model (remove last classification layer)
        # Get the second-to-last layer for feature extraction
        self.cnn_feature_extractor = Model(
            inputs=self.cnn_model.input,
            outputs=self.cnn_model.layers[-2].output
        )
        print(f"CNN model loaded. Feature vector size: {self.cnn_feature_extractor.output.shape[-1]}")
        
    def build_lstm_model(self):
        """Build LSTM model for temporal features"""
        lstm_model = Sequential([
            LSTM(64, activation='relu', return_sequences=True, 
                 input_shape=(self.sequence_length, self.temporal_features)),
            Dropout(0.3),
            LSTM(32, activation='relu'),
            Dropout(0.3),
            Dense(16, activation='relu')
        ])
        return lstm_model
    
    def build_hybrid_model(self):
        """Build hybrid model combining CNN and LSTM"""
        # Load or create CNN feature extractor
        if self.cnn_model is None:
            self.load_cnn_model()
        
        # Image input branch (CNN)
        image_input = Input(shape=(224, 224, 3), name='image_input')
        cnn_features = self.cnn_feature_extractor(image_input)
        
        # Temporal data input branch (LSTM)
        temporal_input = Input(shape=(self.sequence_length, self.temporal_features), 
                              name='temporal_input')
        lstm_model = self.build_lstm_model()
        lstm_features = lstm_model(temporal_input)
        
        # Combine both branches
        combined = Concatenate()([cnn_features, lstm_features])
        
        # Classification head
        x = Dense(128, activation='relu')(combined)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.3)(x)
        output = Dense(self.num_classes, activation='softmax', name='output')(x)
        
        # Create final model
        self.hybrid_model = Model(
            inputs=[image_input, temporal_input],
            outputs=output
        )
        
        # Compile model
        self.hybrid_model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("\n=== Hybrid Model Architecture ===")
        self.hybrid_model.summary()
        
        return self.hybrid_model
    
    def prepare_temporal_features(self, features_dict):
        """
        Prepare temporal features for LSTM
        features_dict should contain: moisture, temperature, pH, nitrogen, 
                                      phosphorus, potassium, organic_matter
        """
        if self.scaler is None and os.path.exists(self.scaler_path):
            with open(self.scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
        elif self.scaler is None:
            self.scaler = StandardScaler()
        
        # Extract features in consistent order
        feature_order = ['moisture', 'temperature', 'pH', 'nitrogen', 
                        'phosphorus', 'potassium', 'organic_matter']
        
        features = []
        for key in feature_order:
            features.append(features_dict.get(key, 0.5))  # Default to 0.5 if missing
        
        # Create sequence by repeating (in production, use actual time-series data)
        sequence = np.tile(features, (self.sequence_length, 1))
        
        # Scale features
        if hasattr(self.scaler, 'mean_'):
            sequence = self.scaler.transform(sequence)
        
        return sequence.reshape(1, self.sequence_length, self.temporal_features)
    
    def predict_hybrid(self, image_array, temporal_features=None):
        """
        Make prediction using hybrid model
        
        Args:
            image_array: Preprocessed image array (224, 224, 3)
            temporal_features: Dict with soil features or None
        
        Returns:
            (prediction_id, confidence, feature_importance)
        """
        if self.hybrid_model is None:
            print("Hybrid model not loaded. Building now...")
            self.build_hybrid_model()
            
            # Try to load weights if they exist
            if os.path.exists(self.lstm_weights_path):
                try:
                    self.hybrid_model.load_weights(self.lstm_weights_path)
                    print("Loaded hybrid model weights")
                except:
                    print("Could not load weights, using untrained model")
        
        # Prepare image
        if len(image_array.shape) == 3:
            image_array = np.expand_dims(image_array, axis=0)
        
        # Prepare temporal features
        if temporal_features is None:
            # Use neutral default values if no temporal data provided
            temporal_features = {
                'moisture': 0.5,
                'temperature': 0.5,
                'pH': 0.5,
                'nitrogen': 0.5,
                'phosphorus': 0.5,
                'potassium': 0.5,
                'organic_matter': 0.5
            }
        
        temporal_array = self.prepare_temporal_features(temporal_features)
        
        # Predict
        predictions = self.hybrid_model.predict(
            [image_array, temporal_array],
            verbose=0
        )
        
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]
        
        return int(predicted_class), float(confidence), predictions[0].tolist()
    
    def predict_cnn_only(self, image_array):
        """Fallback: Use only CNN for prediction"""
        if self.cnn_model is None:
            self.load_cnn_model()
        
        if len(image_array.shape) == 3:
            image_array = np.expand_dims(image_array, axis=0)
        
        predictions = self.cnn_model.predict(image_array, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]
        
        return int(predicted_class), float(confidence), predictions[0].tolist()
    
    def train_hybrid_model(self, X_images, X_temporal, y_labels, 
                          validation_split=0.2, epochs=50, batch_size=16, callbacks=None):
        """
        Train the hybrid model
        
        Args:
            X_images: Array of images (N, 224, 224, 3)
            X_temporal: Array of temporal sequences (N, sequence_length, features)
            y_labels: One-hot encoded labels (N, num_classes)
            callbacks: List of Keras callbacks (optional)
        """
        if self.hybrid_model is None:
            self.build_hybrid_model()
        
        # Fit scaler on temporal data
        n_samples = X_temporal.shape[0]
        temporal_flat = X_temporal.reshape(-1, self.temporal_features)
        self.scaler = StandardScaler()
        temporal_scaled = self.scaler.fit_transform(temporal_flat)
        X_temporal_scaled = temporal_scaled.reshape(n_samples, 
                                                    self.sequence_length, 
                                                    self.temporal_features)
        
        # Save scaler
        with open(self.scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Train model
        from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
        
        if callbacks is None:
            callbacks = [
                ModelCheckpoint(self.lstm_weights_path, 
                              save_best_only=True, 
                              monitor='val_accuracy',
                              mode='max'),
                EarlyStopping(patience=10, restore_best_weights=True)
            ]
        
        history = self.hybrid_model.fit(
            [X_images, X_temporal_scaled],
            y_labels,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        print(f"\nTraining completed! Best validation accuracy: {max(history.history['val_accuracy']):.4f}")
        return history
    
    def save_model(self, path="hybrid_soil_model.h5"):
        """Save the complete hybrid model"""
        if self.hybrid_model is not None:
            self.hybrid_model.save(path)
            print(f"Hybrid model saved to {path}")
    
    def load_hybrid_model(self, path="hybrid_soil_model.h5"):
        """Load complete hybrid model"""
        from tensorflow.keras.layers import DepthwiseConv2D
        
        class CustomDepthwiseConv2D(DepthwiseConv2D):
            def __init__(self, **kwargs):
                kwargs.pop('groups', None)
                super().__init__(**kwargs)
        
        custom_objects = {'DepthwiseConv2D': CustomDepthwiseConv2D}
        self.hybrid_model = load_model(path, custom_objects=custom_objects)
        print(f"Hybrid model loaded from {path}")
        
        # Also load scaler
        if os.path.exists(self.scaler_path):
            with open(self.scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)


if __name__ == "__main__":
    # Test the hybrid model
    print("Testing Hybrid Soil Classification Model...")
    
    classifier = HybridSoilClassifier()
    classifier.build_hybrid_model()
    
    # Test with dummy data
    dummy_image = np.random.rand(224, 224, 3)
    dummy_temporal = {
        'moisture': 0.6,
        'temperature': 0.7,
        'pH': 0.65,
        'nitrogen': 0.5,
        'phosphorus': 0.4,
        'potassium': 0.55,
        'organic_matter': 0.6
    }
    
    pred_id, confidence, all_probs = classifier.predict_hybrid(dummy_image, dummy_temporal)
    
    print(f"\nTest Prediction:")
    print(f"Predicted Class: {pred_id}")
    print(f"Confidence: {confidence:.2%}")
    print(f"All Probabilities: {all_probs}")
