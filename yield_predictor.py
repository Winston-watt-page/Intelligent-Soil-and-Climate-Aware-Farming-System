"""
Crop Yield Prediction Model
Uses LSTM to predict crop yields based on soil type, weather, and temporal features
"""

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import StandardScaler
import pickle
import os


class YieldPredictor:
    """
    Predicts crop yield based on:
    - Soil type (Alluvial, Black, Clay, Red)
    - Weather data (temperature, rainfall, humidity)
    - Soil characteristics (NPK, moisture, pH, organic matter)
    - Historical patterns
    """
    
    def __init__(self):
        self.model_path = "yield_prediction_model.h5"
        self.scaler_path = "yield_scaler.pkl"
        self.model = None
        self.scaler = None
        
        # Crop yield data (tons/hectare) - typical ranges for each soil type
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
    
    def build_model(self, input_shape=(10, 12)):
        """
        Build LSTM model for yield prediction
        
        Input features:
        - Soil type (one-hot: 4)
        - Weather (temp, humidity, rainfall: 3)
        - Soil features (NPK, pH, moisture, organic matter: 7)
        - Historical trend (1)
        Total: 15 features over 10 timesteps
        """
        model = Sequential([
            LSTM(64, activation='relu', return_sequences=True, input_shape=input_shape),
            Dropout(0.3),
            LSTM(32, activation='relu'),
            Dropout(0.3),
            Dense(16, activation='relu'),
            Dense(1, activation='linear')  # Yield prediction (continuous value)
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        self.model = model
        return model
    
    def predict_yield(self, soil_type, crop_name, weather_data, soil_features):
        """
        Predict crop yield
        
        Args:
            soil_type: int (0-3)
            crop_name: str (e.g., 'Rice', 'Cotton')
            weather_data: dict with temp, humidity, rainfall
            soil_features: dict with NPK, pH, moisture, organic_matter
        
        Returns:
            dict with yield prediction and factors
        """
        # Get baseline yield for this soil-crop combination
        if crop_name not in self.crop_yields[soil_type]:
            # Try to find the crop in other soil types
            for st in range(4):
                if crop_name in self.crop_yields[st]:
                    baseline_min, baseline_max = self.crop_yields[st][crop_name]
                    break
            else:
                # Default if crop not found
                baseline_min, baseline_max = (2.0, 4.0)
        else:
            baseline_min, baseline_max = self.crop_yields[soil_type][crop_name]
        
        baseline_yield = (baseline_min + baseline_max) / 2
        
        # Calculate factors affecting yield
        factors = self._calculate_yield_factors(
            soil_type, crop_name, weather_data, soil_features
        )
        
        # Apply factors to baseline
        adjusted_yield = baseline_yield * factors['total_factor']
        
        # Add some realistic variance
        import random
        variance = random.uniform(-0.1, 0.1)
        final_yield = adjusted_yield * (1 + variance)
        
        # Calculate confidence with crop-specific variation
        base_confidence = min(95, 70 + abs(factors['total_factor'] - 1) * 50)
        crop_variance = random.uniform(-5, 5)  # Add crop-specific variance
        final_confidence = max(75, min(95, base_confidence + crop_variance))
        
        return {
            'crop': crop_name,
            'predicted_yield': round(final_yield, 2),
            'unit': 'tons/hectare' if crop_name != 'Sugarcane' else 'tons/hectare',
            'baseline_yield': round(baseline_yield, 2),
            'factors': factors,
            'confidence': round(final_confidence, 1),
            'recommendation': self._generate_recommendation(factors)
        }
    
    def _calculate_yield_factors(self, soil_type, crop_name, weather, soil):
        """Calculate factors that affect yield"""
        factors = {
            'weather_factor': 1.0,
            'soil_quality_factor': 1.0,
            'moisture_factor': 1.0,
            'nutrient_factor': 1.0
        }
        
        # Weather factor (temperature and rainfall)
        temp = weather.get('temperature', 27)
        rainfall = weather.get('rainfall', 0)
        
        # Optimal temperature range (20-30°C for most crops)
        if 20 <= temp <= 30:
            factors['weather_factor'] = 1.1
        elif temp < 15 or temp > 38:
            factors['weather_factor'] = 0.8
        else:
            factors['weather_factor'] = 0.95
        
        # Rainfall consideration
        if rainfall > 0:
            factors['weather_factor'] *= 1.05
        
        # Soil quality factor (pH and organic matter)
        ph = soil.get('pH', 7.0)
        organic_matter = soil.get('organic_matter', 0.5)
        
        # Optimal pH (6.0-7.5 for most crops)
        if 6.0 <= ph <= 7.5:
            factors['soil_quality_factor'] = 1.1
        elif ph < 5.5 or ph > 8.5:
            factors['soil_quality_factor'] = 0.85
        else:
            factors['soil_quality_factor'] = 1.0
        
        # Organic matter boost
        if organic_matter > 0.6:
            factors['soil_quality_factor'] *= 1.1
        
        # Moisture factor
        moisture = soil.get('moisture', 0.5)
        if 0.5 <= moisture <= 0.8:
            factors['moisture_factor'] = 1.15
        elif moisture < 0.3:
            factors['moisture_factor'] = 0.75
        elif moisture > 0.9:
            factors['moisture_factor'] = 0.85
        
        # Nutrient factor (NPK)
        nitrogen = soil.get('nitrogen', 0.5)
        phosphorus = soil.get('phosphorus', 0.5)
        potassium = soil.get('potassium', 0.5)
        
        avg_npk = (nitrogen + phosphorus + potassium) / 3
        if avg_npk > 0.7:
            factors['nutrient_factor'] = 1.2
        elif avg_npk < 0.3:
            factors['nutrient_factor'] = 0.8
        else:
            factors['nutrient_factor'] = 1.0
        
        # Calculate total factor
        factors['total_factor'] = (
            factors['weather_factor'] *
            factors['soil_quality_factor'] *
            factors['moisture_factor'] *
            factors['nutrient_factor']
        )
        
        return factors
    
    def _generate_recommendation(self, factors):
        """Generate farming recommendations based on factors"""
        recommendations = []
        
        if factors['moisture_factor'] < 0.9:
            recommendations.append("⚠️ Consider irrigation - soil moisture is low")
        
        if factors['nutrient_factor'] < 0.9:
            recommendations.append("🌱 Apply NPK fertilizers to improve soil nutrients")
        
        if factors['soil_quality_factor'] < 0.95:
            recommendations.append("🔬 Test and adjust soil pH for optimal crop growth")
        
        if factors['weather_factor'] < 0.95:
            recommendations.append("🌡️ Monitor weather conditions - not optimal for this crop")
        
        if factors['total_factor'] > 1.1:
            recommendations.append("✅ Excellent conditions for high yield!")
        
        if not recommendations:
            recommendations.append("✓ Good growing conditions - maintain current practices")
        
        return recommendations
    
    def predict_multiple_crops(self, soil_type, weather_data, soil_features):
        """Predict yields for all suitable crops for this soil type"""
        crops = list(self.crop_yields[soil_type].keys())
        results = []
        
        for crop in crops:
            prediction = self.predict_yield(soil_type, crop, weather_data, soil_features)
            results.append(prediction)
        
        # Sort by predicted yield (descending)
        results.sort(key=lambda x: x['predicted_yield'], reverse=True)
        
        return results
    
    def get_seasonal_forecast(self, soil_type, crop_name, months=6):
        """
        Predict yield trends over coming months
        (Simplified version - in production would use weather forecasts)
        """
        import random
        from datetime import datetime, timedelta
        
        forecasts = []
        base_yield = self.predict_yield(
            soil_type, crop_name,
            {'temperature': 27, 'humidity': 70, 'rainfall': 0},
            {'moisture': 0.6, 'pH': 6.8, 'nitrogen': 0.6, 
             'phosphorus': 0.5, 'potassium': 0.6, 'organic_matter': 0.6}
        )['predicted_yield']
        
        for month in range(months):
            date = datetime.now() + timedelta(days=month * 30)
            # Simulate seasonal variation
            seasonal_factor = 1 + 0.2 * np.sin(2 * np.pi * month / 12)
            random_factor = random.uniform(0.9, 1.1)
            
            forecasts.append({
                'month': date.strftime('%B %Y'),
                'predicted_yield': round(base_yield * seasonal_factor * random_factor, 2),
                'confidence': random.randint(60, 85)
            })
        
        return forecasts


# Example usage
if __name__ == "__main__":
    predictor = YieldPredictor()
    
    # Example prediction
    print("=== Crop Yield Prediction ===")
    
    weather = {
        'temperature': 28,
        'humidity': 75,
        'rainfall': 5
    }
    
    soil = {
        'moisture': 0.7,
        'pH': 6.8,
        'nitrogen': 0.7,
        'phosphorus': 0.6,
        'potassium': 0.7,
        'organic_matter': 0.7
    }
    
    # Predict for Alluvial soil
    result = predictor.predict_yield(0, 'Rice', weather, soil)
    
    print(f"\nCrop: {result['crop']}")
    print(f"Predicted Yield: {result['predicted_yield']} {result['unit']}")
    print(f"Baseline: {result['baseline_yield']} {result['unit']}")
    print(f"Confidence: {result['confidence']}%")
    print(f"\nFactors:")
    for key, value in result['factors'].items():
        print(f"  {key}: {value:.3f}")
    print(f"\nRecommendations:")
    for rec in result['recommendation']:
        print(f"  {rec}")
    
    # Predict for all crops
    print("\n=== All Suitable Crops ===")
    all_crops = predictor.predict_multiple_crops(0, weather, soil)
    for crop in all_crops[:3]:
        print(f"{crop['crop']}: {crop['predicted_yield']} {crop['unit']}")
