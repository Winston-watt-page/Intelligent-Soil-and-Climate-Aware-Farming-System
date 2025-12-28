from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from flask import Flask, request, render_template, jsonify, session
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.layers import DepthwiseConv2D
import h5py
from hybrid_model import HybridSoilClassifier
from weather_service import WeatherService
from yield_predictor import YieldPredictor
from geolocation_service import GeolocationService
from planting_calendar import PlantingCalendar
from soil_health_monitor import SoilHealthMonitor

import os, sys, glob, re

app = Flask(__name__)
app.secret_key = 'intelligent_farming_system_2025'

# Add zip to Jinja2 templates
app.jinja_env.globals.update(zip=zip)

model_path = "models/soil_classifier_93_86.h5"

# Custom DepthwiseConv2D to handle 'groups' parameter from older models
class CustomDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, **kwargs):
        # Remove 'groups' parameter if present (not supported in new versions)
        kwargs.pop('groups', None)
        super().__init__(**kwargs)

# Load traditional CNN model with custom objects
custom_objects = {'DepthwiseConv2D': CustomDepthwiseConv2D}
model = load_model(model_path, custom_objects=custom_objects, compile=False)

# Initialize hybrid model
print("Initializing Hybrid Soil Classifier...")
hybrid_classifier = HybridSoilClassifier(cnn_model_path=model_path)
try:
    # Build hybrid model structure
    hybrid_classifier.build_hybrid_model()
    
    # Try to load weights if they exist
    if os.path.exists('models/lstm_soil_model.h5'):
        try:
            hybrid_classifier.hybrid_model.load_weights('models/lstm_soil_model.h5')
            print("✓ Hybrid model weights loaded successfully")
        except:
            print("⚠ Could not load hybrid weights, using untrained LSTM")
    else:
        print("⚠ Hybrid model initialized (LSTM not trained yet)")
        print("  Run 'python train_hybrid.py' to train the hybrid model")
except Exception as e:
    print(f"⚠ Could not initialize hybrid model: {e}")
    print("  Falling back to CNN-only predictions")
    hybrid_classifier = None

print("Initializing services...")
weather_service = WeatherService()
yield_predictor = YieldPredictor()
geo_service = GeolocationService()
planting_calendar = PlantingCalendar()
soil_health_monitor = SoilHealthMonitor()
print("✓ All services initialized")

# Bilingual soil classification data (Tamil and English)
soil_data = {
    0: {
        "name_en": "Alluvial Soil",
        "name_ta": "வண்டல் மண்",
        "description_en": "Rich in nutrients and suitable for intensive agriculture",
        "description_ta": "ஊட்டச்சத்துக்கள் நிறைந்த மற்றும் தீவிர விவசாயத்திற்கு ஏற்றது",
        "crops_en": ["Rice", "Wheat", "Sugarcane", "Maize", "Cotton", "Soybean", "Jute"],
        "crops_ta": ["நெல்", "கோதுமை", "கரும்பு", "சோளம்", "பருத்தி", "சோயாபீன்", "சணல்"],
        "primary_crops": ["Rice", "Wheat", "Sugarcane"],
        "secondary_crops": ["Maize", "Cotton", "Soybean", "Jute", "Vegetables"],
        "color": "#8B4513"
    },
    1: {
        "name_en": "Black Soil",
        "name_ta": "கருப்பு மண்",
        "description_en": "High moisture retention capacity, ideal for cotton cultivation",
        "description_ta": "அதிக ஈரப்பதம் தக்கவைக்கும் திறன், பருத்தி பயிருக்கு ஏற்றது",
        "crops_en": ["Cotton", "Wheat", "Jowar", "Millets", "Linseed", "Castor", "Sunflower"],
        "crops_ta": ["பருத்தி", "கோதுமை", "சோளம்", "தினை", "ஆளி விதை", "ஆமணக்கு", "சூரியகாந்தி"],
        "primary_crops": ["Cotton", "Wheat", "Jowar"],
        "secondary_crops": ["Millets", "Linseed", "Castor", "Sunflower", "Groundnut"],
        "color": "#2C1810"
    },
    2: {
        "name_en": "Clay Soil",
        "name_ta": "களிமண்",
        "description_en": "Heavy soil with good water retention, suitable for specific crops",
        "description_ta": "நல்ல நீர் தேக்கும் திறன் கொண்ட கனமான மண்",
        "crops_en": ["Rice", "Lettuce", "Chard", "Broccoli", "Cabbage", "Snap Beans"],
        "crops_ta": ["நெல்", "கீரை", "சார்ட்", "ப்ரோக்கோலி", "முட்டைகோஸ்", "பீன்ஸ்"],
        "primary_crops": ["Rice", "Lettuce", "Cabbage"],
        "secondary_crops": ["Broccoli", "Chard", "Snap Beans", "Cauliflower", "Spinach"],
        "color": "#CD853F"
    },
    3: {
        "name_en": "Red Soil",
        "name_ta": "சிவப்பு மண்",
        "description_en": "Rich in iron, good drainage, suitable for diverse crops",
        "description_ta": "இரும்புச்சத்து நிறைந்த, நல்ல வடிகால், பல்வேறு பயிர்களுக்கு ஏற்றது",
        "crops_en": ["Cotton", "Wheat", "Pulses", "Millets", "Oil Seeds", "Potatoes"],
        "crops_ta": ["பருத்தி", "கோதுமை", "பருப்பு வகைகள்", "தினை", "எண்ணெய் விதைகள்", "உருளைக்கிழங்கு"],
        "primary_crops": ["Cotton", "Millets", "Pulses"],
        "secondary_crops": ["Wheat", "Groundnut", "Potatoes", "Oil Seeds", "Tobacco"],
        "color": "#A0522D"
    }
}

def model_predict(image_path, model, temporal_features=None, use_hybrid=True):
    """
    Predict soil type using hybrid model (CNN + LSTM) or CNN only
    
    Args:
        image_path: Path to soil image
        model: CNN model (fallback)
        temporal_features: Optional dict with soil features
        use_hybrid: Whether to use hybrid model if available
    
    Returns:
        result_id, soil_info, confidence, all_probabilities
    """
    print("Predicting soil type...")
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = image / 255
    
    # Try hybrid prediction first
    if use_hybrid and hybrid_classifier is not None:
        try:
            result, confidence, all_probs = hybrid_classifier.predict_hybrid(
                image, temporal_features
            )
            print(f"✓ Hybrid prediction: {soil_data[result]['name_en']} (confidence: {confidence:.2%})")
            return result, soil_data[result], confidence, all_probs
        except Exception as e:
            print(f"⚠ Hybrid prediction failed: {e}. Using CNN only.")
    
    # Fallback to CNN-only prediction
    image = np.expand_dims(image, axis=0)
    predictions = model.predict(image, verbose=0)
    result = np.argmax(predictions[0])
    confidence = float(predictions[0][result])
    all_probs = predictions[0].tolist()
    
    # Convert numpy int64 to Python int for JSON serialization
    result = int(result)
    print(f"✓ CNN prediction: {soil_data[result]['name_en']} (confidence: {confidence:.2%})")
    return result, soil_data[result], confidence, all_probs


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/api/get-location', methods=['GET'])
def get_location():
    """API endpoint to auto-detect user's location"""
    try:
        location = geo_service.get_current_location()
        return jsonify(location)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/reverse-geocode', methods=['POST'])
def reverse_geocode():
    """API endpoint to convert coordinates to location"""
    try:
        data = request.get_json()
        lat = data.get('latitude')
        lon = data.get('longitude')
        
        if not lat or not lon:
            return jsonify({'error': 'Missing coordinates'}), 400
        
        location = geo_service.get_location_by_coords(lat, lon)
        return jsonify(location)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/weekly-forecast', methods=['GET'])
def get_weekly_forecast():
    """API endpoint to get 7-day weather forecast"""
    try:
        city = request.args.get('city')
        lat = request.args.get('lat')
        lon = request.args.get('lon')
        
        if city:
            forecast = weather_service.get_weekly_forecast(city_name=city)
        elif lat and lon:
            forecast = weather_service.get_weekly_forecast(lat=float(lat), lon=float(lon))
        else:
            forecast = weather_service.get_weekly_forecast(city_name='Chennai')
        
        return jsonify(forecast)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            file = request.files.get('image')
            if not file:
                return jsonify({'error': 'No image uploaded'}), 400
            
            filename = secure_filename(file.filename)
            file_path = os.path.join('static/user uploaded', filename)
            
            # Ensure directory exists
            os.makedirs('static/user uploaded', exist_ok=True)
            file.save(file_path)
            
            # Get location (auto-detect or manual)
            location = None
            city = request.form.get('city', '')
            detected_lat = request.form.get('detected_lat')
            detected_lon = request.form.get('detected_lon')
            
            # Priority 1: Use detected GPS coordinates if available
            if detected_lat and detected_lon:
                try:
                    lat = float(detected_lat)
                    lon = float(detected_lon)
                    location = geo_service.get_location_by_coords(lat, lon)
                    city = location['city']
                    print(f"Using GPS coordinates: {city} ({lat}, {lon})")
                except Exception as e:
                    print(f"Error using GPS coordinates: {e}")
                    location = None
            
            # Priority 2: Use selected city from dropdown
            if not location and city and city != 'auto' and city != 'manual':
                # Use manually selected city
                location = {
                    'city': city,
                    'latitude': None,
                    'longitude': None,
                    'source': 'manual_city'
                }
                print(f"Using selected city: {city}")
            
            # Priority 3: Auto-detect via IP
            if not location or city == 'auto':
                # Auto-detect location via IP
                location = geo_service.get_current_location()
                city = location['city']
                print(f"Auto-detected location via IP: {city} ({location['latitude']}, {location['longitude']})")
            
            # Standard soil type values (normalized 0-1 scale)
            soil_type_standards = {
                0: {  # Alluvial Soil
                    'moisture': 0.60,
                    'pH': 6.8,
                    'nitrogen': 0.60,
                    'phosphorus': 0.55,
                    'potassium': 0.60,
                    'organic_matter': 0.60
                },
                1: {  # Black Soil (Regur)
                    'moisture': 0.70,
                    'pH': 7.6,
                    'nitrogen': 0.45,
                    'phosphorus': 0.40,
                    'potassium': 0.75,
                    'organic_matter': 0.55
                },
                2: {  # Clay Soil
                    'moisture': 0.80,
                    'pH': 7.2,
                    'nitrogen': 0.55,
                    'phosphorus': 0.50,
                    'potassium': 0.70,
                    'organic_matter': 0.55
                },
                3: {  # Red Soil
                    'moisture': 0.45,
                    'pH': 6.2,
                    'nitrogen': 0.40,
                    'phosphorus': 0.40,
                    'potassium': 0.55,
                    'organic_matter': 0.40
                }
            }
            
            # Extract temporal features if provided (optional)
            temporal_features = None
            user_provided_temporal = False
            
            try:
                # Check if temporal features are provided in the form
                if request.form.get('use_temporal') == 'true':
                    user_provided_temporal = True
                    temporal_features = {
                        'moisture': float(request.form.get('moisture', 0.5)),
                        'temperature': float(request.form.get('temperature', 25)),
                        'pH': float(request.form.get('pH', 7.0)),
                        'nitrogen': float(request.form.get('nitrogen', 0.5)),
                        'phosphorus': float(request.form.get('phosphorus', 0.5)),
                        'potassium': float(request.form.get('potassium', 0.5)),
                        'organic_matter': float(request.form.get('organic_matter', 0.5))
                    }
                    print(f"✓ Using user-provided temporal features: {temporal_features}")
            except Exception as e:
                print(f"Could not parse temporal features: {e}")
                temporal_features = None
                user_provided_temporal = False
            
            # If hybrid model is requested but no temporal features provided,
            # first predict soil type with CNN only, then use standard values
            if request.form.get('use_temporal') == 'true' and not user_provided_temporal:
                print("ℹ User requested hybrid model but didn't provide temporal features")
                print("→ Step 1: Predicting soil type with CNN only...")
                
                # CNN-only prediction to determine soil type
                initial_result_id, _, _, _ = model_predict(
                    file_path, model, temporal_features=None, use_hybrid=False
                )
                
                # Use standard values for the predicted soil type
                temporal_features = soil_type_standards[initial_result_id].copy()
                temporal_features['temperature'] = 25  # Default temperature
                
                print(f"→ Step 2: Using standard values for {soil_data[initial_result_id]['name_en']}")
                print(f"  Standard values: {temporal_features}")
                user_provided_temporal = False  # Mark as auto-generated
            
            # Predict soil type (with or without temporal features)
            result_id, soil_info, confidence, all_probs = model_predict(
                file_path, model, temporal_features, 
                use_hybrid=(temporal_features is not None)
            )
            
            # Prepare probability data for visualization
            soil_names = ['Alluvial', 'Black', 'Clay', 'Red']
            probabilities = [
                {'name': soil_names[i], 'prob': float(all_probs[i])} 
                for i in range(len(all_probs))
            ]
            
            # Get current weather data
            weather_data = weather_service.get_weather_by_city(city)
            
            # Get 7-day weather forecast
            if location and location.get('latitude') and location.get('longitude'):
                weekly_forecast = weather_service.get_weekly_forecast(
                    lat=location['latitude'], 
                    lon=location['longitude']
                )
            else:
                weekly_forecast = weather_service.get_weekly_forecast(city_name=city)
            
            # Prepare LSTM weather data
            lstm_weather_data = weather_service.prepare_lstm_weather_data(weekly_forecast)
            
            # Prepare soil features for yield prediction
            # Use temporal features if available (user-provided or auto-generated)
            # Otherwise use standard values for the predicted soil type
            if temporal_features:
                soil_features = {k: v for k, v in temporal_features.items() if k != 'temperature'}
            else:
                # Use standard values for the predicted soil type
                soil_features = soil_type_standards[result_id].copy()
            
            # Add weather temp to soil features
            weather_for_yield = {
                'temperature': weather_data.get('temperature', 27),
                'humidity': weather_data.get('humidity', 70),
                'rainfall': weekly_forecast[0].get('total_rain', 0) if weekly_forecast else 0
            }
            
            # Get primary and secondary crop recommendations
            primary_crops = soil_info.get('primary_crops', soil_info['crops_en'][:3])
            secondary_crops = soil_info.get('secondary_crops', soil_info['crops_en'][3:])
            
            # Predict yields for primary crops
            primary_crop_yields = []
            for crop in primary_crops[:3]:  # Top 3 primary crops
                yield_pred = yield_predictor.predict_yield(
                    result_id, crop, weather_for_yield, soil_features
                )
                primary_crop_yields.append(yield_pred)
            
            # Predict yields for secondary crops
            secondary_crop_yields = []
            for crop in secondary_crops[:3]:  # Top 3 secondary crops
                yield_pred = yield_predictor.predict_yield(
                    result_id, crop, weather_for_yield, soil_features
                )
                secondary_crop_yields.append(yield_pred)
            
            # Get planting calendar for primary crops
            planting_schedules = planting_calendar.get_full_calendar(
                result_id, primary_crops[:3]
            )
            
            # Get current planting recommendations
            current_plantable = planting_calendar.get_current_recommendations(result_id)
            
            # Determine model type description
            if temporal_features and user_provided_temporal:
                model_description = 'Hybrid CNN+LSTM (User-provided values)'
            elif temporal_features and not user_provided_temporal:
                model_description = f'Hybrid CNN+LSTM (Standard {soil_info["name_en"]} values)'
            else:
                model_description = 'CNN Only'
            
            # Render results
            return render_template('result.html', 
                                 soil_info=soil_info,
                                 image_path=file_path,
                                 soil_id=result_id,
                                 confidence=confidence,
                                 probabilities=probabilities,
                                 used_temporal=temporal_features is not None,
                                 user_provided_temporal=user_provided_temporal,
                                 model_type=model_description,
                                 weather=weather_data,
                                 weekly_forecast=weekly_forecast,
                                 location=location,
                                 primary_crops=primary_crop_yields,
                                 secondary_crops=secondary_crop_yields,
                                 lstm_enabled=True,
                                 soil_features=soil_features)
        
        except Exception as e:
            print(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': str(e)}), 500


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/features')
def features():
    return render_template('features.html')


@app.route('/weather/<city>')
def get_weather(city):
    """API endpoint to get weather data for a city"""
    try:
        weather_data = weather_service.get_weather_by_city(city)
        return jsonify(weather_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/yield-prediction', methods=['POST'])
def predict_yield():
    """API endpoint for yield prediction"""
    try:
        data = request.json
        soil_type = int(data.get('soil_type', 0))
        crop_name = data.get('crop', 'Rice')
        
        weather_data = data.get('weather', {
            'temperature': 27,
            'humidity': 70,
            'rainfall': 0
        })
        
        soil_features = data.get('soil_features', {
            'moisture': 0.6,
            'pH': 6.8,
            'nitrogen': 0.6,
            'phosphorus': 0.5,
            'potassium': 0.6,
            'organic_matter': 0.6
        })
        
        prediction = yield_predictor.predict_yield(
            soil_type, crop_name, weather_data, soil_features
        )
        
        return jsonify(prediction)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True,threaded=False)
