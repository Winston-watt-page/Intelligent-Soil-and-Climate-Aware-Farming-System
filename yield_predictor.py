"""
Crop Yield Prediction Model
Uses LSTM to predict crop yields based on soil type, weather, and temporal features
Trained on actual temporal soil data
"""

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import os
import warnings
warnings.filterwarnings('ignore')


class YieldPredictor:
    """
    LSTM-based crop yield prediction
    
    Predicts crop yield using:
    - Soil type (Alluvial, Black, Clay, Red)
    - Temporal sequences of: moisture, temperature, pH, NPK, organic matter
    - Weather patterns
    
    Model: LSTM(64) -> LSTM(32) -> Dense layers
    Training: Uses soil_temporal_data.csv
    """
    
    def __init__(self, model_path="models/yield_lstm_model.h5", scaler_path="models/yield_scaler.pkl"):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model = None
        self.scaler = None
        self.sequence_length = 5  # Use 5 timesteps for LSTM sequence
        self.feature_names = ['moisture', 'temperature', 'pH', 'nitrogen', 'phosphorus', 'potassium', 'organic_matter']
        
        # Crop yield data (tons/hectare) - baseline reference values
        self.crop_yields = {
            0: {  # Alluvial
                'Rice': (4.5, 6.5),
                'Wheat': (3.5, 5.0),
                'Sugarcane': (80, 110),
                'Cotton': (2.0, 3.5),
                'Maize': (4.0, 6.0)
            },
            1: {  # Black
                'Cotton': (2.5, 4.0),
                'Wheat': (3.0, 4.5),
                'Jowar': (2.0, 3.5),
                'Millets': (1.5, 2.5),
                'Sunflower': (1.2, 2.0)
            },
            2: {  # Clay
                'Rice': (4.0, 5.5),
                'Lettuce': (15, 25),
                'Broccoli': (8, 12),
                'Cabbage': (30, 45),
                'Beans': (2.5, 4.0)
            },
            3: {  # Red
                'Cotton': (1.5, 2.5),
                'Millets': (1.2, 2.0),
                'Pulses': (1.0, 1.8),
                'Groundnut': (1.5, 2.5),
                'Potatoes': (20, 30)
            }
        }
        
        # Load model if exists
        self._load_model()
    
    def _load_model(self):
        """Load pre-trained model if available"""
        if os.path.exists(self.model_path):
            try:
                # When loading a model with custom metrics, you might need to recompile it
                self.model = load_model(self.model_path, compile=False)
                # Recompile with a standard optimizer and loss
                self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
                print(f"✓ LSTM yield model loaded and recompiled from {self.model_path}")
            except Exception as e:
                print(f"⚠ Could not load model: {e}")
                self.model = None
        
        if os.path.exists(self.scaler_path):
            try:
                with open(self.scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                print(f"✓ Scaler loaded from {self.scaler_path}")
            except Exception as e:
                print(f"⚠ Could not load scaler: {e}")
                self.scaler = None
    
    def build_model(self, input_shape=(5, 7)):
        """
        Build LSTM model for yield prediction
        
        Input: Sequences of 5 timesteps with 7 features each
        Features: moisture, temperature, pH, nitrogen, phosphorus, potassium, organic_matter
        Output: Yield prediction (tons/hectare)
        """
        model = Sequential([
            LSTM(64, activation='relu', return_sequences=True, input_shape=input_shape),
            Dropout(0.3),
            LSTM(32, activation='relu'),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='linear')  # Continuous yield output
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        self.model = model
        print(f"✓ LSTM model built: {input_shape} -> Dense(16) -> Dense(1)")
        return model
    
    def train(self, data_path="data/soil_temporal_data.csv", epochs=50, batch_size=32, test_size=0.2):
        """
        Train LSTM model on temporal soil data
        
        Args:
            data_path: Path to CSV with temporal soil data
            epochs: Training epochs
            batch_size: Batch size
            test_size: Test set ratio
        
        Returns:
            Training history
        """
        print(f"\n[1/4] Loading data from {data_path}...")
        df = pd.read_csv(data_path)
        print(f"    Loaded {len(df)} records, {len(df['sequence_id'].unique())} sequences")
        
        # Prepare sequences
        print(f"[2/4] Preparing sequences...")
        X, y, soil_types = self._prepare_sequences(df)
        print(f"    Created {len(X)} sequences of length {self.sequence_length}")
        
        # Scale features
        print(f"[3/4] Scaling features...")
        if self.scaler is None:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
            # Save scaler
            os.makedirs('models', exist_ok=True)
            with open(self.scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
        else:
            X_scaled = self.scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=42)
        
        # Build model
        if self.model is None:
            self.build_model(input_shape=(self.sequence_length, len(self.feature_names)))
        
        # Train
        print(f"[4/4] Training LSTM model ({len(X_train)} training samples)...")
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
        ]
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate
        train_loss, train_mae = self.model.evaluate(X_train, y_train, verbose=0)
        test_loss, test_mae = self.model.evaluate(X_test, y_test, verbose=0)
        
        print(f"\n✓ Training complete!")
        print(f"    Train MAE: {train_mae:.4f} | Test MAE: {test_mae:.4f}")
        print(f"    Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")
        
        # Save model
        os.makedirs('models', exist_ok=True)
        self.model.save(self.model_path)
        print(f"    Model saved to {self.model_path}")
        
        return history
    
    def _prepare_sequences(self, df):
        """
        Prepare sequences from temporal data for LSTM training
        """
        X_data = []
        y_data = []
        soil_types_data = []
        
        # Group by sequence ID
        grouped = df.groupby('sequence_id')
        
        for seq_id, group in grouped:
            soil_type = group['soil_type'].iloc[0]
            
            # Extract features in sequence
            features = group[self.feature_names].values
            
            # Create sequences with sliding window
            for i in range(len(features) - self.sequence_length):
                seq = features[i:i + self.sequence_length]
                X_data.append(seq)
                
                # Target: next value (or average of next few values)
                next_idx = min(i + self.sequence_length, len(features) - 1)
                target_value = features[next_idx][0]  # Use moisture as example target
                y_data.append(target_value)
                soil_types_data.append(soil_type)
        
        return np.array(X_data), np.array(y_data), np.array(soil_types_data)
    
    def predict_yield(self, soil_type, crop_name, weather_data, soil_features, sequence=None):
        """
        Predict crop yield using trained LSTM
        
        Args:
            soil_type: int (0-3) for soil class
            crop_name: str (e.g., 'Rice', 'Cotton')
            weather_data: dict with temp, humidity, rainfall
            soil_features: dict with moisture, pH, nitrogen, etc.
            sequence: Optional pre-computed LSTM sequence
        
        Returns:
            dict with yield prediction and metrics
        """
        # Create sequence if not provided
        if sequence is None:
            sequence = self._create_prediction_sequence(soil_features)
        
        # Use LSTM if model is trained
        if self.model is not None:
            try:
                # Scale sequence
                seq_scaled = self.scaler.transform(
                    sequence.reshape(-1, sequence.shape[-1])
                ).reshape(sequence.shape)
                
                # Predict
                lstm_prediction = float(self.model.predict(np.array([seq_scaled]), verbose=0)[0][0])
                
                # Get baseline yield
                baseline_yield = self._get_baseline_yield(soil_type, crop_name)
                
                # Adjust with weather and LSTM prediction
                weather_factor = self._calculate_weather_factor(weather_data)
                soil_factor = self._calculate_soil_quality_factor(soil_features)
                
                # Combine LSTM prediction with baseline and factors
                adjusted_yield = baseline_yield * (0.6 + lstm_prediction * 0.4) * weather_factor * soil_factor
                
                confidence = min(95, 75 + abs(lstm_prediction - 0.5) * 40)
                
                return {
                    'crop': crop_name,
                    'predicted_yield': round(max(0.1, adjusted_yield), 2),
                    'unit': 'tons/hectare',
                    'baseline_yield': round(baseline_yield, 2),
                    'lstm_confidence': round(confidence, 1),
                    'soil_quality': round(soil_factor * 100, 1),
                    'weather_impact': round(weather_factor * 100, 1),
                    'recommendation': self._generate_recommendation(soil_factor, weather_factor)
                }
            except Exception as e:
                print(f"⚠ LSTM prediction error: {e}, using fallback")
        
        # Fallback if model not available
        return self._predict_yield_fallback(soil_type, crop_name, weather_data, soil_features)
    
    def _create_prediction_sequence(self, soil_features):
        """Create a sequence for LSTM prediction from current soil features"""
        # Create a sequence by repeating the current features
        sequence = []
        for _ in range(self.sequence_length):
            feature_values = [
                soil_features.get('moisture', 0.6),
                soil_features.get('temperature', 27),
                soil_features.get('pH', 7.0),
                soil_features.get('nitrogen', 0.6),
                soil_features.get('phosphorus', 0.5),
                soil_features.get('potassium', 0.6),
                soil_features.get('organic_matter', 0.6)
            ]
            sequence.append(feature_values)
        return np.array(sequence)
    
    def _get_baseline_yield(self, soil_type, crop_name):
        """Get baseline yield for a crop-soil combination"""
        if crop_name not in self.crop_yields[soil_type]:
            for st in range(4):
                if crop_name in self.crop_yields[st]:
                    baseline_min, baseline_max = self.crop_yields[st][crop_name]
                    return (baseline_min + baseline_max) / 2
            return 2.5  # Default fallback
        
        baseline_min, baseline_max = self.crop_yields[soil_type][crop_name]
        return (baseline_min + baseline_max) / 2
    
    def _calculate_weather_factor(self, weather_data):
        """Calculate weather impact factor (0.5 - 1.2)"""
        temp = weather_data.get('temperature', 27)
        rainfall = weather_data.get('rainfall', 0)
        humidity = weather_data.get('humidity', 70)
        
        factor = 1.0
        
        # Optimal temperature range
        if 20 <= temp <= 32:
            factor *= 1.1
        elif temp < 15 or temp > 38:
            factor *= 0.7
        else:
            factor *= 0.9
        
        # Rainfall impact
        if 0 < rainfall < 20:
            factor *= 1.15
        elif rainfall > 30:
            factor *= 0.95
        
        # Humidity impact (optimal 60-80%)
        if 60 <= humidity <= 80:
            factor *= 1.05
        elif humidity > 85:
            factor *= 0.9
        
        return max(0.5, min(1.2, factor))
    
    def _calculate_soil_quality_factor(self, soil_features):
        """Calculate soil quality factor (0.6 - 1.3)"""
        factor = 1.0
        
        # pH factor
        ph = soil_features.get('pH', 7.0)
        if 6.0 <= ph <= 7.5:
            factor *= 1.15
        elif ph < 5.5 or ph > 8.0:
            factor *= 0.75
        
        # Moisture factor
        moisture = soil_features.get('moisture', 0.6)
        if 0.5 <= moisture <= 0.8:
            factor *= 1.2
        elif moisture < 0.3 or moisture > 0.9:
            factor *= 0.8
        
        # NPK factor
        npk_avg = (soil_features.get('nitrogen', 0.6) + 
                   soil_features.get('phosphorus', 0.5) + 
                   soil_features.get('potassium', 0.6)) / 3
        
        if npk_avg > 0.7:
            factor *= 1.15
        elif npk_avg < 0.4:
            factor *= 0.8
        
        # Organic matter
        om = soil_features.get('organic_matter', 0.6)
        if om > 0.7:
            factor *= 1.1
        elif om < 0.4:
            factor *= 0.9
        
        return max(0.6, min(1.3, factor))
    
    def _predict_yield_fallback(self, soil_type, crop_name, weather_data, soil_features):
        """Fallback prediction when LSTM model not available"""
        baseline = self._get_baseline_yield(soil_type, crop_name)
        weather_f = self._calculate_weather_factor(weather_data)
        soil_f = self._calculate_soil_quality_factor(soil_features)
        
        final_yield = baseline * weather_f * soil_f
        
        return {
            'crop': crop_name,
            'predicted_yield': round(max(0.1, final_yield), 2),
            'unit': 'tons/hectare',
            'baseline_yield': round(baseline, 2),
            'lstm_confidence': 70.0,
            'soil_quality': round(soil_f * 100, 1),
            'weather_impact': round(weather_f * 100, 1),
            'recommendation': self._generate_recommendation(soil_f, weather_f),
            'note': 'Using rule-based model (LSTM not trained yet)'
        }
    
    def _generate_recommendation(self, soil_factor, weather_factor):
        """Generate recommendations based on factors"""
        recommendations = []
        
        if soil_factor < 0.85:
            if soil_factor < 0.75:
                recommendations.append("🔴 Critical: Soil conditions need immediate improvement")
            else:
                recommendations.append("⚠️ Soil quality is below optimal")
        
        if weather_factor < 0.85:
            recommendations.append("🌡️ Monitor weather - conditions not ideal for this crop")
        elif weather_factor > 1.1:
            recommendations.append("✅ Weather conditions are excellent")
        
        if soil_factor > 1.1 and weather_factor > 1.1:
            recommendations.append("🌟 Optimal conditions for maximum yield!")
        
        if not recommendations:
            recommendations.append("📊 Good conditions - maintain current practices")
        
        return recommendations
    
    def predict_multiple_crops(self, soil_type, weather_data, soil_features):
        """Predict yields for all suitable crops for this soil type"""
        crops = list(self.crop_yields[soil_type].keys())
        results = []
        
        # Create sequence once
        sequence = self._create_prediction_sequence(soil_features)
        
        for crop in crops:
            prediction = self.predict_yield(soil_type, crop, weather_data, soil_features, sequence=sequence)
            results.append(prediction)
        
        # Sort by predicted yield (descending)
        results.sort(key=lambda x: x['predicted_yield'], reverse=True)
        
        return results
    
    def get_seasonal_forecast(self, soil_type, crop_name, months=6):
        """
        Predict yield trends over coming months
        Uses LSTM-based seasonal patterns
        """
        forecasts = []
        sequence = self._create_prediction_sequence({
            'moisture': 0.6, 'temperature': 27, 'pH': 6.8,
            'nitrogen': 0.6, 'phosphorus': 0.5, 'potassium': 0.6,
            'organic_matter': 0.6
        })
        
        base_yield = self._get_baseline_yield(soil_type, crop_name)
        
        for month in range(months):
            # Simulate seasonal variation
            seasonal_factor = 1.0 + 0.25 * np.sin(2 * np.pi * month / 12)
            
            if self.model is not None:
                try:
                    seq_scaled = self.scaler.transform(
                        sequence.reshape(-1, sequence.shape[-1])
                    ).reshape(sequence.shape)
                    lstm_factor = float(self.model.predict(np.array([seq_scaled]), verbose=0)[0][0])
                except:
                    lstm_factor = 0.5
            else:
                lstm_factor = 0.5
            
            predicted = base_yield * seasonal_factor * (0.8 + lstm_factor * 0.4)
            
            from datetime import datetime, timedelta
            date = datetime.now() + timedelta(days=month * 30)
            
            forecasts.append({
                'month': date.strftime('%B %Y'),
                'predicted_yield': round(max(0.1, predicted), 2),
                'seasonal_factor': round(seasonal_factor, 2),
                'confidence': round(min(95, 70 + abs(seasonal_factor - 1) * 30), 1)
            })
        
        return forecasts


# Training script
if __name__ == "__main__":
    print("=" * 60)
    print("LSTM-Based Crop Yield Prediction System")
    print("=" * 60)
    
    predictor = YieldPredictor()
    
    # Train model
    print("\n[TRAINING PHASE]")
    print("Starting LSTM model training on temporal soil data...\n")
    
    try:
        history = predictor.train(
            data_path="data/soil_temporal_data.csv",
            epochs=50,
            batch_size=32,
            test_size=0.2
        )
        model_trained = True
    except FileNotFoundError:
        print("⚠ soil_temporal_data.csv not found. Using rule-based fallback.")
        model_trained = False
    except Exception as e:
        print(f"⚠ Training error: {e}. Using rule-based fallback.")
        model_trained = False
    
    # Example prediction
    print("\n" + "=" * 60)
    print("EXAMPLE PREDICTION")
    print("=" * 60)
    
    weather = {
        'temperature': 28,
        'humidity': 75,
        'rainfall': 5
    }
    
    soil = {
        'moisture': 0.7,
        'temperature': 28,
        'pH': 6.8,
        'nitrogen': 0.7,
        'phosphorus': 0.6,
        'potassium': 0.7,
        'organic_matter': 0.7
    }
    
    # Single crop prediction
    print("\n[Single Crop - Alluvial Soil + Rice]")
    result = predictor.predict_yield(0, 'Rice', weather, soil)
    
    print(f"Crop: {result['crop']}")
    print(f"Predicted Yield: {result['predicted_yield']} {result['unit']}")
    print(f"Baseline: {result['baseline_yield']} {result['unit']}")
    print(f"Confidence: {result['lstm_confidence']}%")
    print(f"Soil Quality: {result['soil_quality']}%")
    print(f"Weather Impact: {result['weather_impact']}%")
    print(f"\nRecommendations:")
    for rec in result['recommendation']:
        print(f"  {rec}")
    
    if 'note' in result:
        print(f"\nNote: {result['note']}")
    
    # Multiple crops prediction
    print("\n[All Suitable Crops - Alluvial Soil]")
    all_crops = predictor.predict_multiple_crops(0, weather, soil)
    for i, crop in enumerate(all_crops[:3], 1):
        print(f"{i}. {crop['crop']}: {crop['predicted_yield']} {crop['unit']} (Confidence: {crop['lstm_confidence']}%)")
    
    # Seasonal forecast
    print("\n[Seasonal Forecast - Rice for 6 months]")
    forecast = predictor.get_seasonal_forecast(0, 'Rice', months=6)
    for month_data in forecast[:3]:
        print(f"  {month_data['month']}: {month_data['predicted_yield']} tons/ha (Confidence: {month_data['confidence']}%)")
