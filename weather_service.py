"""
Real-time Weather Integration Module
Fetches current weather data and forecasts for soil analysis
"""

import requests
from datetime import datetime
import json

class WeatherService:
    """
    Integrates with OpenWeatherMap API for real-time weather data
    Free API key required: https://openweathermap.org/api
    """
    
    def __init__(self, api_key=None):
        self.api_key = api_key or "d079e59a8d4c7af83e5763ab6f3e70fd"  # OpenWeatherMap API key
        self.base_url = "https://api.openweathermap.org/data/2.5"
        self.city_profiles = {
            'chennai': (29, 75),
            'tambaram': (29, 74),
            'pallavaram': (29, 74),
            'tiruchirappalli': (30, 72),
            'trichy': (30, 72),
            'thottiyam': (30, 72),
            'lalgudi': (30, 72),
            'musiri': (30, 72),
            'srirangam': (30, 72),
            'manachanallur': (30, 72),
            'manapparai': (31, 68),
            'thuvarankurichi': (30, 69),
            'namakkal': (28, 67),
            'mohanur': (28, 67),
            'rasipuram': (28, 66),
            'tiruchengode': (28, 67),
            'paramathi velur': (29, 68),
            'kolli hills': (22, 70),
            'coimbatore': (26, 66),
            'pollachi': (27, 67),
            'mettupalayam': (25, 68),
            'valparai': (21, 80),
            'salem': (28, 68),
            'attur': (29, 69),
            'mettur': (28, 70),
            'omalur': (28, 68),
            'erode': (27, 66),
            'gobichettipalayam': (27, 68),
            'bhavani': (27, 67),
            'perundurai': (27, 67),
            'madurai': (31, 68),
            'melur': (31, 67),
            'usilampatti': (32, 66),
            'vadipatti': (31, 68),
            'thanjavur': (30, 73),
            'kumbakonam': (30, 73),
            'pattukkottai': (31, 74),
            'thiruvidaimarudur': (30, 73),
            'tiruppur': (27, 67),
            'nilgiris': (18, 78),
            'ooty': (18, 78),
            'dindigul': (30, 69),
            'theni': (31, 70),
            'virudhunagar': (32, 69),
            'sivaganga': (32, 70),
            'ramanathapuram': (31, 74),
            'tiruvarur': (30, 74),
            'nagapattinam': (29, 76),
            'pudukkottai': (31, 74),
            'ariyalur': (30, 72),
            'dharmapuri': (29, 67),
            'krishnagiri': (28, 66),
            'vellore': (29, 70),
            'tiruvannamalai': (30, 69),
            'ranipet': (29, 70),
            'tirupattur': (28, 69),
            'cuddalore': (29, 76),
            'villupuram': (30, 75),
            'kallakurichi': (30, 74),
            'kanchipuram': (29, 75),
            'chengalpattu': (29, 75),
            'tirunelveli': (32, 74),
            'thoothukudi': (31, 76),
            'tenkasi': (30, 72),
            'kanniyakumari': (30, 78),
            'karur': (29, 68),
            'perambalur': (30, 72)
        }

    def _get_city_profile(self, city_name):
        normalized = (city_name or '').strip().lower()
        if normalized in self.city_profiles:
            return self.city_profiles[normalized]

        if 'trichy' in normalized or 'tiruchirappalli' in normalized:
            return self.city_profiles['tiruchirappalli']
        if 'chennai' in normalized:
            return self.city_profiles['chennai']
        if 'coimbatore' in normalized:
            return self.city_profiles['coimbatore']
        if 'madurai' in normalized:
            return self.city_profiles['madurai']
        if 'salem' in normalized:
            return self.city_profiles['salem']
        if 'erode' in normalized:
            return self.city_profiles['erode']
        if 'thanjavur' in normalized:
            return self.city_profiles['thanjavur']
        if 'tirunelveli' in normalized:
            return self.city_profiles['tirunelveli']
        if 'tuticorin' in normalized or 'thoothukudi' in normalized:
            return self.city_profiles['thoothukudi']

        return (27, 70)
        
    def get_weather_by_coords(self, lat, lon):
        """
        Get current weather by coordinates
        
        Args:
            lat: Latitude
            lon: Longitude
        
        Returns:
            dict with weather data
        """
        try:
            if self.api_key == "demo":
                return self._get_demo_weather(lat, lon)
            
            url = f"{self.base_url}/weather"
            params = {
                'lat': lat,
                'lon': lon,
                'appid': self.api_key,
                'units': 'metric'
            }
            
            response = requests.get(url, params=params, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                return self._parse_weather_data(data)
            else:
                print(f"Weather API error: {response.status_code}")
                return self._get_demo_weather(lat, lon)
                
        except Exception as e:
            print(f"Error fetching weather: {e}")
            return self._get_demo_weather(lat, lon)
    
    def get_weather_by_city(self, city_name, country_code="IN"):
        """
        Get current weather by city name
        
        Args:
            city_name: City name (e.g., "Chennai", "Coimbatore")
            country_code: Country code (default: "IN" for India)
        
        Returns:
            dict with weather data
        """
        try:
            if self.api_key == "demo":
                return self._get_demo_weather_city(city_name)
            
            url = f"{self.base_url}/weather"
            params = {
                'q': f"{city_name},{country_code}",
                'appid': self.api_key,
                'units': 'metric'
            }
            
            response = requests.get(url, params=params, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                return self._parse_weather_data(data)
            else:
                return self._get_demo_weather_city(city_name)
                
        except Exception as e:
            print(f"Error fetching weather: {e}")
            return self._get_demo_weather_city(city_name)
    
    def _parse_weather_data(self, data):
        """Parse OpenWeatherMap API response"""
        return {
            'temperature': data['main']['temp'],
            'feels_like': data['main']['feels_like'],
            'humidity': data['main']['humidity'],
            'pressure': data['main']['pressure'],
            'description': data['weather'][0]['description'],
            'icon': data['weather'][0]['icon'],
            'wind_speed': data['wind']['speed'],
            'clouds': data['clouds']['all'],
            'location': data['name'],
            'country': data['sys']['country'],
            'timestamp': datetime.fromtimestamp(data['dt']).strftime('%Y-%m-%d %H:%M:%S'),
            'sunrise': datetime.fromtimestamp(data['sys']['sunrise']).strftime('%H:%M'),
            'sunset': datetime.fromtimestamp(data['sys']['sunset']).strftime('%H:%M')
        }
    
    def _get_demo_weather(self, lat, lon):
        """Return demo weather data when API key not available"""
        import random
        
        # Simulate realistic weather for Tamil Nadu region
        return {
            'temperature': round(25 + random.uniform(-5, 10), 1),
            'feels_like': round(27 + random.uniform(-5, 10), 1),
            'humidity': random.randint(60, 85),
            'pressure': random.randint(1008, 1015),
            'description': random.choice(['clear sky', 'few clouds', 'scattered clouds', 'light rain']),
            'icon': '01d',
            'wind_speed': round(random.uniform(2, 8), 1),
            'clouds': random.randint(10, 60),
            'location': f'Location ({lat:.2f}, {lon:.2f})',
            'country': 'IN',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'sunrise': '06:15',
            'sunset': '18:30',
            'demo_mode': True
        }
    
    def _get_demo_weather_city(self, city_name):
        """Return demo weather data for a city"""
        import random

        base_temp, base_humidity = self._get_city_profile(city_name)
        
        return {
            'temperature': round(base_temp + random.uniform(-3, 3), 1),
            'feels_like': round(base_temp + random.uniform(-2, 5), 1),
            'humidity': base_humidity + random.randint(-10, 10),
            'pressure': random.randint(1008, 1015),
            'description': random.choice(['partly cloudy', 'clear sky', 'scattered clouds']),
            'icon': '02d',
            'wind_speed': round(random.uniform(3, 10), 1),
            'clouds': random.randint(20, 50),
            'location': city_name,
            'country': 'IN',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'sunrise': '06:15',
            'sunset': '18:30',
            'demo_mode': True
        }
    
    def get_forecast(self, city_name, country_code="IN", days=5):
        """
        Get weather forecast
        
        Args:
            city_name: City name
            country_code: Country code
            days: Number of days (max 5 for free API)
        
        Returns:
            list of forecast data
        """
        try:
            if self.api_key == "demo":
                return self._get_demo_forecast(city_name, days)
            
            url = f"{self.base_url}/forecast"
            params = {
                'q': f"{city_name},{country_code}",
                'appid': self.api_key,
                'units': 'metric',
                'cnt': days * 8  # 8 forecasts per day (3-hour intervals)
            }
            
            response = requests.get(url, params=params, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                return self._parse_forecast_data(data)
            else:
                return self._get_demo_forecast(city_name, days)
                
        except Exception as e:
            print(f"Error fetching forecast: {e}")
            return self._get_demo_forecast(city_name, days)
    
    def _parse_forecast_data(self, data):
        """Parse forecast API response"""
        forecasts = []
        for item in data['list']:
            forecasts.append({
                'date': datetime.fromtimestamp(item['dt']).strftime('%Y-%m-%d %H:%M'),
                'temperature': item['main']['temp'],
                'humidity': item['main']['humidity'],
                'description': item['weather'][0]['description'],
                'wind_speed': item['wind']['speed'],
                'rain': item.get('rain', {}).get('3h', 0)
            })
        return forecasts
    
    def _get_demo_forecast(self, city_name, days):
        """Return demo forecast data"""
        import random
        from datetime import timedelta
        
        forecasts = []
        base_temp, _ = self._get_city_profile(city_name)
        
        for day in range(days):
            date = datetime.now() + timedelta(days=day)
            for hour in [6, 12, 18]:
                forecasts.append({
                    'date': date.replace(hour=hour).strftime('%Y-%m-%d %H:%M'),
                    'temperature': round(base_temp + random.uniform(-5, 8), 1),
                    'humidity': random.randint(60, 85),
                    'description': random.choice(['clear', 'clouds', 'rain']),
                    'wind_speed': round(random.uniform(3, 12), 1),
                    'rain': round(random.uniform(0, 5), 1) if random.random() > 0.7 else 0,
                    'demo_mode': True
                })
        
        return forecasts
    
    def get_weekly_forecast(self, city_name=None, lat=None, lon=None):
        """
        Get 7-day weather forecast with daily summaries
        
        Args:
            city_name: City name (optional if lat/lon provided)
            lat: Latitude (optional if city_name provided)
            lon: Longitude (optional if city_name provided)
        
        Returns:
            dict with daily forecast summaries for 7 days
        """
        from datetime import timedelta
        
        try:
            # Get detailed forecast
            if city_name:
                forecast_data = self.get_forecast(city_name, days=7)
            elif lat and lon:
                forecast_data = self._get_forecast_by_coords(lat, lon, days=7)
            else:
                return self._get_demo_weekly_forecast()
            
            # Group by day and calculate daily summaries
            daily_forecasts = {}
            
            for item in forecast_data:
                date_str = item['date'].split()[0]  # Get just the date part
                
                if date_str not in daily_forecasts:
                    daily_forecasts[date_str] = {
                        'temps': [],
                        'humidity': [],
                        'rain': [],
                        'wind': [],
                        'descriptions': []
                    }
                
                daily_forecasts[date_str]['temps'].append(item['temperature'])
                daily_forecasts[date_str]['humidity'].append(item['humidity'])
                daily_forecasts[date_str]['rain'].append(item.get('rain', 0))
                daily_forecasts[date_str]['wind'].append(item['wind_speed'])
                daily_forecasts[date_str]['descriptions'].append(item['description'])
            
            # Calculate daily averages
            weekly_forecast = []
            for date_str in sorted(daily_forecasts.keys())[:7]:
                data = daily_forecasts[date_str]
                
                weekly_forecast.append({
                    'date': date_str,
                    'day_name': datetime.strptime(date_str, '%Y-%m-%d').strftime('%A'),
                    'temp_min': round(min(data['temps']), 1),
                    'temp_max': round(max(data['temps']), 1),
                    'temp_avg': round(sum(data['temps']) / len(data['temps']), 1),
                    'humidity_avg': round(sum(data['humidity']) / len(data['humidity'])),
                    'total_rain': round(sum(data['rain']), 1),
                    'wind_avg': round(sum(data['wind']) / len(data['wind']), 1),
                    'description': max(set(data['descriptions']), key=data['descriptions'].count)
                })
            
            return weekly_forecast
            
        except Exception as e:
            print(f"Error getting weekly forecast: {e}")
            return self._get_demo_weekly_forecast()
    
    def _get_forecast_by_coords(self, lat, lon, days=7):
        """Get forecast by coordinates"""
        try:
            if self.api_key == "demo":
                return self._get_demo_forecast("Location", days)
            
            url = f"{self.base_url}/forecast"
            params = {
                'lat': lat,
                'lon': lon,
                'appid': self.api_key,
                'units': 'metric',
                'cnt': days * 8
            }
            
            response = requests.get(url, params=params, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                return self._parse_forecast_data(data)
            else:
                return self._get_demo_forecast("Location", days)
                
        except Exception as e:
            print(f"Error fetching forecast by coords: {e}")
            return self._get_demo_forecast("Location", days)
    
    def _get_demo_weekly_forecast(self):
        """Generate demo 7-day forecast"""
        import random
        from datetime import timedelta
        
        weekly_forecast = []
        base_temp = 27
        
        for day in range(7):
            date = datetime.now() + timedelta(days=day)
            temp_variation = random.uniform(-3, 5)
            
            weekly_forecast.append({
                'date': date.strftime('%Y-%m-%d'),
                'day_name': date.strftime('%A'),
                'temp_min': round(base_temp + temp_variation - 2, 1),
                'temp_max': round(base_temp + temp_variation + 4, 1),
                'temp_avg': round(base_temp + temp_variation, 1),
                'humidity_avg': random.randint(60, 85),
                'total_rain': round(random.uniform(0, 15), 1) if random.random() > 0.6 else 0,
                'wind_avg': round(random.uniform(3, 12), 1),
                'description': random.choice(['clear sky', 'few clouds', 'scattered clouds', 'light rain', 'moderate rain']),
                'demo_mode': True
            })
        
        return weekly_forecast
    
    def prepare_lstm_weather_data(self, weekly_forecast):
        """
        Prepare weather forecast data for LSTM input
        
        Args:
            weekly_forecast: List of 7-day forecast data
        
        Returns:
            numpy array shaped for LSTM (sequence_length, features)
        """
        import numpy as np
        
        # Extract features for each day
        lstm_data = []
        for day in weekly_forecast[:7]:  # Ensure we have 7 days
            features = [
                day['temp_avg'] / 40.0,  # Normalize temperature (0-40°C range)
                day['humidity_avg'] / 100.0,  # Normalize humidity (0-100%)
                min(day['total_rain'] / 50.0, 1.0),  # Normalize rainfall (0-50mm range)
                day['wind_avg'] / 20.0,  # Normalize wind speed (0-20 m/s range)
            ]
            lstm_data.append(features)
        
        # Pad if less than 7 days
        while len(lstm_data) < 7:
            lstm_data.append([0.675, 0.7, 0.0, 0.25])  # Default values
        
        return np.array(lstm_data)

# Example usage
if __name__ == "__main__":
    weather = WeatherService()
    
    # Test with demo mode
    print("=== Weather for Chennai ===")
    data = weather.get_weather_by_city("Chennai")
    print(f"Temperature: {data['temperature']}°C")
    print(f"Humidity: {data['humidity']}%")
    print(f"Description: {data['description']}")
    print(f"Wind Speed: {data['wind_speed']} m/s")
    
    print("\n=== 5-Day Forecast ===")
    forecast = weather.get_forecast("Chennai", days=3)
    for f in forecast[:5]:
        print(f"{f['date']}: {f['temperature']}°C, {f['description']}")
    
    print("\n=== 7-Day Weekly Forecast ===")
    weekly = weather.get_weekly_forecast("Chennai")
    for day in weekly:
        print(f"{day['day_name']} ({day['date']}): {day['temp_min']}°C - {day['temp_max']}°C, "
              f"Rain: {day['total_rain']}mm, {day['description']}")
    
    print("\n=== LSTM Weather Data ===")
    lstm_data = weather.prepare_lstm_weather_data(weekly)
    print(f"Shape: {lstm_data.shape}")
    print(f"Sample data (first 3 days):\n{lstm_data[:3]}")

