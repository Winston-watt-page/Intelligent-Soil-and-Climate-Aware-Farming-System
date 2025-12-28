"""
Advanced Training Script for Hybrid Soil Classification
Includes data augmentation and improved accuracy techniques
"""

import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
import os
import sys
import glob
sys.path.append('..')
from hybrid_model import HybridSoilClassifier

def augment_images(images, labels, augmentation_factor=5):
    """
    Apply data augmentation to increase dataset size and variety
    
    Args:
        images: Original images
        labels: Original labels
        augmentation_factor: Number of augmented versions per image
    
    Returns:
        Augmented images and labels
    """
    print(f"Applying data augmentation (factor={augmentation_factor})...")
    
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )
    
    augmented_images = []
    augmented_labels = []
    
    # Keep original images
    augmented_images.extend(images)
    augmented_labels.extend(labels)
    
    # Generate augmented versions
    for img, label in zip(images, labels):
        img_reshaped = img.reshape((1,) + img.shape)
        
        aug_iter = datagen.flow(img_reshaped, batch_size=1)
        
        for i in range(augmentation_factor):
            aug_img = next(aug_iter)[0]
            augmented_images.append(aug_img)
            augmented_labels.append(label)
    
    print(f"Dataset expanded: {len(images)} → {len(augmented_images)} images")
    
    return np.array(augmented_images), np.array(augmented_labels)

def load_soil_images_enhanced(data_dir='soil_dataset', target_size=(224, 224)):
    """
    Enhanced image loading with better preprocessing
    """
    images = []
    labels = []
    
    soil_types = {'Alluvial': 0, 'Black': 1, 'Clay': 2, 'Red': 3}
    
    if not os.path.exists(data_dir):
        print(f"Warning: {data_dir} not found. Creating synthetic training data...")
        # Create realistic dummy data with more samples
        for label_idx in range(4):
            for i in range(200):  # Increased from 50 to 200
                # Create more realistic random patterns
                dummy_img = np.random.rand(*target_size, 3)
                # Add some texture
                noise = np.random.normal(0, 0.1, (*target_size, 3))
                dummy_img = np.clip(dummy_img + noise, 0, 1)
                images.append(dummy_img)
                labels.append(label_idx)
        
        print(f"Created {len(images)} synthetic images")
        return np.array(images), np.array(labels)
    
    for soil_name, label in soil_types.items():
        soil_path = os.path.join(data_dir, soil_name)
        if not os.path.exists(soil_path):
            print(f"Skipping {soil_name} - directory not found")
            continue
            
        image_files = glob.glob(os.path.join(soil_path, '*.jpg')) + \
                     glob.glob(os.path.join(soil_path, '*.png')) + \
                     glob.glob(os.path.join(soil_path, '*.jpeg'))
        
        print(f"Loading images for {soil_name}: {len(image_files)} files found")
        
        for img_path in image_files:
            try:
                img = load_img(img_path, target_size=target_size)
                img_array = img_to_array(img) / 255.0
                images.append(img_array)
                labels.append(label)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
    
    if len(images) == 0:
        print("No images found in dataset. Creating synthetic data...")
        return load_soil_images_enhanced()  # Fall back to synthetic data
    
    return np.array(images), np.array(labels)

def prepare_temporal_data_enhanced(csv_path='../data/soil_temporal_data.csv', sequence_length=5):
    """
    Enhanced temporal data preparation
    """
    if not os.path.exists(csv_path):
        print(f"Temporal data not found at {csv_path}")
        print("Generating new temporal data...")
        os.system('python generate_soil_data.py')
        
        # Check again
        if not os.path.exists(csv_path):
            csv_path = 'soil_temporal_data.csv'  # Try alternate path
    
    df = pd.read_csv(csv_path)
    
    sequences = []
    labels = []
    
    for (soil_type, seq_id), group in df.groupby(['soil_type', 'sequence_id']):
        if len(group) == sequence_length:
            features = group[['moisture', 'temperature', 'pH', 'nitrogen', 
                            'phosphorus', 'potassium', 'organic_matter']].values
            sequences.append(features)
            labels.append(soil_type)
    
    return np.array(sequences), np.array(labels)

def main():
    print("=" * 70)
    print("    ADVANCED HYBRID SOIL CLASSIFICATION - HIGH ACCURACY TRAINING")
    print("=" * 70)
    
    # Step 1: Load and preprocess images
    print("\n[1/6] Loading soil images...")
    X_images, y_image_labels = load_soil_images_enhanced()
    print(f"✓ Loaded {len(X_images)} base images")
    
    # Step 2: Apply data augmentation for better accuracy
    print("\n[2/6] Applying data augmentation...")
    X_images_aug, y_labels_aug = augment_images(X_images, y_image_labels, augmentation_factor=3)
    
    # Step 3: Load temporal data
    print("\n[3/6] Loading temporal features...")
    X_temporal, y_temporal_labels = prepare_temporal_data_enhanced()
    print(f"✓ Loaded {len(X_temporal)} temporal sequences")
    
    # Step 4: Align datasets
    print("\n[4/6] Aligning datasets...")
    min_samples = min(len(X_images_aug), len(X_temporal))
    
    # Expand temporal data to match augmented images
    if len(X_temporal) < len(X_images_aug):
        # Repeat temporal sequences to match image count
        repeat_factor = len(X_images_aug) // len(X_temporal) + 1
        X_temporal_expanded = np.tile(X_temporal, (repeat_factor, 1, 1))[:len(X_images_aug)]
        y_temporal_expanded = np.tile(y_temporal_labels, repeat_factor)[:len(X_images_aug)]
        X_temporal = X_temporal_expanded
        y_labels = y_labels_aug
    else:
        X_images_aug = X_images_aug[:len(X_temporal)]
        y_labels = y_labels_aug[:len(X_temporal)]
    
    # Shuffle data
    indices = np.random.permutation(len(X_images_aug))
    X_images_final = X_images_aug[indices]
    X_temporal_final = X_temporal[indices]
    y_labels_final = y_labels[indices]
    
    # One-hot encode
    y_categorical = to_categorical(y_labels_final, num_classes=4)
    
    print(f"✓ Final dataset: {len(X_images_final)} samples")
    print(f"  Label distribution: {np.bincount(y_labels_final.astype(int))}")
    
    # Step 5: Setup callbacks for better training
    print("\n[5/6] Setting up training callbacks...")
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            '../models/best_hybrid_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Step 6: Train with improved hyperparameters
    print("\n[6/6] Training hybrid model (optimized for accuracy)...")
    print("  • Epochs: 50 (with early stopping)")
    print("  • Batch size: 8 (better gradients)")
    print("  • Learning rate: adaptive")
    print("  • Data augmentation: enabled")
    
    classifier = HybridSoilClassifier()
    
    history = classifier.train_hybrid_model(
        X_images_final,
        X_temporal_final,
        y_categorical,
        validation_split=0.2,
        epochs=50,
        batch_size=8,
        callbacks=callbacks
    )
    
    # Save final model
    print("\nSaving models...")
    classifier.save_model('../models/hybrid_soil_model.h5')
    
    # Plot training history
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(14, 5))
    
    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    plt.title('Model Accuracy (Improved Training)', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    plt.title('Model Loss (Improved Training)', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../results/training_history_improved.png', dpi=150, bbox_inches='tight')
    print("✓ Training plots saved to '../results/training_history_improved.png'")
    
    # Print summary
    print("\n" + "=" * 70)
    print("    TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    
    best_val_acc = max(history.history['val_accuracy'])
    final_val_acc = history.history['val_accuracy'][-1]
    
    print(f"\n📊 Training Results:")
    print(f"  • Best Validation Accuracy: {best_val_acc:.2%}")
    print(f"  • Final Validation Accuracy: {final_val_acc:.2%}")
    print(f"  • Total Epochs Trained: {len(history.history['accuracy'])}")
    print(f"  • Training Samples: {len(X_images_final)}")
    
    print(f"\n💾 Models Saved:")
    print(f"  • models/hybrid_soil_model.h5")
    print(f"  • models/lstm_soil_model.h5 (weights)")
    print(f"  • models/soil_feature_scaler.pkl")
    print(f"  • models/best_hybrid_model.h5 (best checkpoint)")
    
    print(f"\n🚀 Improvements Applied:")
    print(f"  ✓ Data augmentation (3x dataset expansion)")
    print(f"  ✓ Early stopping (prevent overfitting)")
    print(f"  ✓ Learning rate reduction (adaptive learning)")
    print(f"  ✓ Model checkpointing (save best weights)")
    print(f"  ✓ Optimized batch size (better gradients)")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
