"""
Generate synthetic soil data for LSTM training
Includes temporal features: moisture, temperature, pH, NPK, organic matter
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

# Soil type characteristics (based on domain knowledge)
soil_characteristics = {
    0: {  # Alluvial Soil
        'name': 'Alluvial',
        'moisture': (0.6, 0.8),
        'temperature': (20, 30),
        'pH': (6.5, 7.5),
        'nitrogen': (0.6, 0.9),
        'phosphorus': (0.5, 0.8),
        'potassium': (0.6, 0.9),
        'organic_matter': (0.7, 0.9)
    },
    1: {  # Black Soil
        'name': 'Black',
        'moisture': (0.7, 0.9),  # High water retention
        'temperature': (25, 35),
        'pH': (7.2, 8.5),  # Slightly alkaline
        'nitrogen': (0.5, 0.7),
        'phosphorus': (0.4, 0.6),
        'potassium': (0.7, 0.9),
        'organic_matter': (0.5, 0.7)
    },
    2: {  # Clay Soil
        'name': 'Clay',
        'moisture': (0.7, 0.85),  # Good water retention
        'temperature': (18, 28),
        'pH': (6.0, 7.0),
        'nitrogen': (0.4, 0.6),
        'phosphorus': (0.3, 0.5),
        'potassium': (0.5, 0.7),
        'organic_matter': (0.4, 0.6)
    },
    3: {  # Red Soil
        'name': 'Red',
        'moisture': (0.4, 0.6),  # Lower water retention
        'temperature': (22, 32),
        'pH': (5.5, 6.5),  # Acidic
        'nitrogen': (0.3, 0.5),
        'phosphorus': (0.3, 0.5),
        'potassium': (0.4, 0.6),
        'organic_matter': (0.3, 0.5)
    }
}

def generate_temporal_sequence(soil_type, num_sequences=50, sequence_length=5):
    """
    Generate temporal sequences for a specific soil type
    """
    data = []
    
    for seq_id in range(num_sequences):
        # Base values for this soil type
        char = soil_characteristics[soil_type]
        
        # Generate a sequence with some temporal variation
        for timestep in range(sequence_length):
            # Add temporal variation (e.g., seasonal changes)
            seasonal_factor = np.sin(2 * np.pi * timestep / sequence_length) * 0.1
            
            record = {
                'soil_type': soil_type,
                'soil_name': char['name'],
                'sequence_id': seq_id,
                'timestep': timestep,
                'moisture': np.clip(np.random.uniform(*char['moisture']) + seasonal_factor, 0, 1),
                'temperature': np.random.uniform(*char['temperature']),
                'pH': np.random.uniform(*char['pH']),
                'nitrogen': np.clip(np.random.uniform(*char['nitrogen']) + seasonal_factor * 0.5, 0, 1),
                'phosphorus': np.clip(np.random.uniform(*char['phosphorus']), 0, 1),
                'potassium': np.clip(np.random.uniform(*char['potassium']), 0, 1),
                'organic_matter': np.clip(np.random.uniform(*char['organic_matter']) + seasonal_factor * 0.3, 0, 1)
            }
            data.append(record)
    
    return data

def main():
    """Generate complete dataset for all soil types"""
    
    all_data = []
    
    # Generate data for each soil type
    for soil_type in range(4):
        print(f"Generating data for {soil_characteristics[soil_type]['name']} soil...")
        soil_data = generate_temporal_sequence(soil_type, num_sequences=200, sequence_length=5)
        all_data.extend(soil_data)
    
    # Create DataFrame
    df = pd.DataFrame(all_data)
    
    # Save to CSV
    output_file = 'soil_temporal_data.csv'
    df.to_csv(output_file, index=False)
    
    print(f"\nDataset created successfully!")
    print(f"Total records: {len(df)}")
    print(f"File saved: {output_file}")
    print(f"\nDataset shape: {df.shape}")
    print(f"\nFirst few rows:")
    print(df.head(10))
    print(f"\nSoil type distribution:")
    print(df.groupby('soil_name')['sequence_id'].nunique())
    print(f"\nFeature statistics:")
    print(df[['moisture', 'temperature', 'pH', 'nitrogen', 'phosphorus', 'potassium', 'organic_matter']].describe())

if __name__ == "__main__":
    main()
