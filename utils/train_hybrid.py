"""
Train the Hybrid Soil Classification Model
Combines CNN (images) + LSTM (temporal features)
"""

import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import os
import glob
import sys
sys.path.append('..')  # Add parent directory to path
from hybrid_model import HybridSoilClassifier
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_soil_images(data_dir='soil_dataset', target_size=(224, 224), augment=True):
    """
    Load soil images from directory structure:
    soil_dataset/
        Alluvial/
        Black/
        Clay/
        Red/
    
    Args:
        augment: If True, apply data augmentation to increase dataset
    """
    images = []
    labels = []
    
    soil_types = {'Alluvial': 0, 'Black': 1, 'Clay': 2, 'Red': 3}
    
    if not os.path.exists(data_dir):
        print(f"Warning: {data_dir} not found. Will create dummy data for testing.")
        # Create dummy data
        for label_idx in range(4):
            for i in range(50):
                # Random image
                dummy_img = np.random.rand(*target_size, 3) * 255
                images.append(dummy_img / 255.0)
                labels.append(label_idx)
        return np.array(images), np.array(labels)
    
    for soil_name, label in soil_types.items():
        soil_path = os.path.join(data_dir, soil_name)
        if not os.path.exists(soil_path):
            print(f"Skipping {soil_name} - directory not found")
            continue
            
        image_files = glob.glob(os.path.join(soil_path, '*.jpg')) + \
                     glob.glob(os.path.join(soil_path, '*.png'))
        
        print(f"Loading {len(image_files)} images for {soil_name}...")
        
        for img_path in image_files[:100]:  # Limit to 100 images per class
            try:
                img = load_img(img_path, target_size=target_size)
                img_array = img_to_array(img) / 255.0
                images.append(img_array)
                labels.append(label)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
    
    return np.array(images), np.array(labels)

def prepare_temporal_data(csv_path='../data/soil_temporal_data.csv', sequence_length=5):
    """
    Prepare temporal data from CSV
    """
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found. Generating data...")
        # Generate data
        os.system('python generate_soil_data.py')
    
    df = pd.read_csv(csv_path)
    
    # Group by sequence
    sequences = []
    labels = []
    
    for (soil_type, seq_id), group in df.groupby(['soil_type', 'sequence_id']):
        if len(group) == sequence_length:
            # Extract features in order
            features = group[['moisture', 'temperature', 'pH', 'nitrogen', 
                            'phosphorus', 'potassium', 'organic_matter']].values
            sequences.append(features)
            labels.append(soil_type)
    
    return np.array(sequences), np.array(labels)

def main():
    print("=" * 60)
    print("HYBRID SOIL CLASSIFICATION MODEL - TRAINING")
    print("=" * 60)
    
    # Step 1: Generate temporal data if needed
    if not os.path.exists('soil_temporal_data.csv'):
        print("\n[1/5] Generating temporal soil data...")
        os.system('python generate_soil_data.py')
    else:
        print("\n[1/5] Temporal data already exists")
    
    # Step 2: Load images
    print("\n[2/5] Loading soil images...")
    X_images, y_image_labels = load_soil_images()
    print(f"Loaded {len(X_images)} images")
    print(f"Image shape: {X_images[0].shape}")
    
    # Step 3: Load temporal data
    print("\n[3/5] Loading temporal features...")
    X_temporal, y_temporal_labels = prepare_temporal_data()
    print(f"Loaded {len(X_temporal)} temporal sequences")
    print(f"Temporal shape: {X_temporal[0].shape}")
    
    # Step 4: Align datasets
    print("\n[4/5] Aligning datasets...")
    min_samples = min(len(X_images), len(X_temporal))
    X_images = X_images[:min_samples]
    X_temporal = X_temporal[:min_samples]
    y_labels = y_image_labels[:min_samples]
    
    # Shuffle
    indices = np.random.permutation(min_samples)
    X_images = X_images[indices]
    X_temporal = X_temporal[indices]
    y_labels = y_labels[indices]
    
    # One-hot encode labels
    y_categorical = to_categorical(y_labels, num_classes=4)
    
    print(f"Final dataset size: {min_samples} samples")
    print(f"Label distribution: {np.bincount(y_labels)}")
    
    # Step 5: Train hybrid model with improved settings
    print("\n[5/5] Training hybrid model with improved hyperparameters...")
    classifier = HybridSoilClassifier()
    
    # Use better training parameters for higher accuracy
    history = classifier.train_hybrid_model(
        X_images, 
        X_temporal, 
        y_categorical,
        validation_split=0.2,
        epochs=50,  # More epochs for better convergence
        batch_size=8   # Smaller batch size for better gradients
    )
    
    # Save the model to models folder
    classifier.save_model('../models/hybrid_soil_model.h5')
    
    # Plot training history
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('../results/training_history.png', dpi=150, bbox_inches='tight')
    print("\nTraining plots saved to '../results/training_history.png'")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"Best validation accuracy: {max(history.history['val_accuracy']):.2%}")
    print("Models saved:")
    print("  - models/hybrid_soil_model.h5")
    print("  - models/lstm_soil_model.h5 (weights)")
    print("  - models/soil_feature_scaler.pkl")

if __name__ == "__main__":
    main()
