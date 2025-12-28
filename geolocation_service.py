"""
Geolocation Service
Automatically detects user's location based on IP address
"""

import requests
import json


class GeolocationService:
    """
    Provides automatic geolocation detection using IP-based services
    Falls back to default location if detection fails
    """
    
    def __init__(self):
        # Free geolocation APIs
        self.ipapi_url = "http://ip-api.com/json/"
        self.ipinfo_url = "https://ipinfo.io/json"
        
        # Default location (Chennai, Tamil Nadu)
        self.default_location = {
            'latitude': 13.0827,
            'longitude': 80.2707,
            'city': 'Chennai',
            'region': 'Tamil Nadu',
            'country': 'India',
            'country_code': 'IN',
            'source': 'default'
        }
    
    def get_current_location(self):
        """
        Automatically detect user's current location
        
        Returns:
            dict with latitude, longitude, city, region, country
        """
        # Try ip-api.com first (no API key needed)
        try:
            response = requests.get(self.ipapi_url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'success':
                    # Check if it's a local/private IP
                    detected_lat = data.get('lat')
                    detected_lon = data.get('lon')
                    
                    # If coordinates are 0,0 or missing, it's likely localhost
                    if detected_lat and detected_lon and (detected_lat != 0 or detected_lon != 0):
                        return {
                            'latitude': detected_lat,
                            'longitude': detected_lon,
                            'city': data.get('city', self.default_location['city']),
                            'region': data.get('regionName', self.default_location['region']),
                            'country': data.get('country', self.default_location['country']),
                            'country_code': data.get('countryCode', self.default_location['country_code']),
                            'zip': data.get('zip', ''),
                            'timezone': data.get('timezone', 'Asia/Kolkata'),
                            'isp': data.get('isp', 'Unknown'),
                            'source': 'ip-api.com'
                        }
        except Exception as e:
            print(f"IP-API failed: {e}")
        
        # Fallback to default location
        print("WARNING: IP geolocation unavailable (localhost). Using default location (Chennai)")
        print("TIP: Use browser geolocation or deploy to get accurate location")
        return self.default_location
    
    def get_location_by_coords(self, lat, lon):
        """
        Reverse geocoding: Get location details from coordinates
        Uses multiple APIs for better reliability
        
        Args:
            lat: Latitude
            lon: Longitude
        
        Returns:
            dict with location details
        """
        print(f"Reverse geocoding coordinates: {lat}, {lon}")
        
        # Method 1: Try BigDataCloud (free, no API key, reliable)
        try:
            url = "https://api.bigdatacloud.net/data/reverse-geocode-client"
            params = {
                'latitude': lat,
                'longitude': lon,
                'localityLanguage': 'en'
            }
            
            response = requests.get(url, params=params, timeout=5)
            if response.status_code == 200:
                data = response.json()
                city = data.get('city') or data.get('locality') or data.get('principalSubdivision', 'Unknown')
                region = data.get('principalSubdivision', 'Unknown')
                country = data.get('countryName', 'Unknown')
                country_code = data.get('countryCode', 'UN')
                
                print(f"BigDataCloud: {city}, {region}, {country}")
                
                return {
                    'latitude': float(lat),
                    'longitude': float(lon),
                    'city': city,
                    'region': region,
                    'country': country,
                    'country_code': country_code,
                    'display_name': f"{city}, {region}, {country}",
                    'source': 'bigdatacloud'
                }
        except Exception as e:
            print(f"WARNING: BigDataCloud failed: {e}")
        
        # Method 2: Try Nominatim (OpenStreetMap)
        try:
            url = "https://nominatim.openstreetmap.org/reverse"
            params = {
                'lat': lat,
                'lon': lon,
                'format': 'json',
                'addressdetails': 1,
                'zoom': 10
            }
            headers = {
                'User-Agent': 'SoilClassificationApp/1.0 (Educational Project)'
            }
            
            response = requests.get(url, params=params, headers=headers, timeout=5)
            if response.status_code == 200:
                data = response.json()
                address = data.get('address', {})
                
                city = (address.get('city') or 
                       address.get('town') or 
                       address.get('village') or 
                       address.get('municipality') or
                       address.get('county', 'Unknown'))
                region = address.get('state', 'Unknown')
                country = address.get('country', 'Unknown')
                country_code = address.get('country_code', 'UN').upper()
                
                print(f"Nominatim: {city}, {region}, {country}")
                
                return {
                    'latitude': float(lat),
                    'longitude': float(lon),
                    'city': city,
                    'region': region,
                    'country': country,
                    'country_code': country_code,
                    'display_name': data.get('display_name', f'{city}, {region}'),
                    'source': 'nominatim'
                }
        except Exception as e:
            print(f"WARNING: Nominatim failed: {e}")
        
        # Method 3: Try geocode.xyz (free API)
        try:
            url = "https://geocode.xyz/{},{}".format(lat, lon)
            params = {'json': 1}
            
            response = requests.get(url, params=params, timeout=5)
            if response.status_code == 200:
                data = response.json()
                if 'city' in data:
                    city = data.get('city', 'Unknown')
                    region = data.get('state', 'Unknown')
                    country = data.get('country', 'Unknown')
                    
                    print(f"Geocode.xyz: {city}, {region}, {country}")
                    
                    return {
                        'latitude': float(lat),
                        'longitude': float(lon),
                        'city': city,
                        'region': region,
                        'country': country,
                        'country_code': 'IN',  # Assume India for this project
                        'display_name': f"{city}, {region}, {country}",
                        'source': 'geocode.xyz'
                    }
        except Exception as e:
            print(f"WARNING: Geocode.xyz failed: {e}")
        
        # Fallback: Return coordinates only
        print(f"WARNING: All geocoding services failed, returning coordinates only")
        return {
            'latitude': float(lat),
            'longitude': float(lon),
            'city': f'Location ({round(lat, 2)}, {round(lon, 2)})',
            'region': 'Unknown',
            'country': 'Unknown',
            'country_code': 'UN',
            'display_name': f'{lat}, {lon}',
            'source': 'coords_only'
        }
    
    def validate_coordinates(self, lat, lon):
        """
        Validate latitude and longitude values
        
        Args:
            lat: Latitude (-90 to 90)
            lon: Longitude (-180 to 180)
        
        Returns:
            bool: True if valid
        """
        try:
            lat = float(lat)
            lon = float(lon)
            return -90 <= lat <= 90 and -180 <= lon <= 180
        except:
            return False
    
    def get_location_info(self):
        """
        Get comprehensive location information including:
        - Current location (auto-detected)
        - Nearby cities
        - Agricultural region info
        
        Returns:
            dict with comprehensive location data
        """
        location = self.get_current_location()
        
        # Add agricultural context based on region
        agricultural_info = self._get_agricultural_context(
            location['city'], 
            location['region']
        )
        
        location['agricultural_zone'] = agricultural_info
        
        return location
    
    def _get_agricultural_context(self, city, region):
        """
        Get agricultural zone information for the region
        
        Returns:
            dict with zone info
        """
        # Comprehensive Tamil Nadu agricultural zones (33 districts)
        tamil_nadu_zones = {
            # Capital and Metro
            'Chennai': {
                'zone': 'Coastal Plain',
                'soil_types': ['Alluvial', 'Red', 'Clay'],
                'common_crops': ['Rice', 'Groundnut', 'Sugarcane']
            },
            'Tambaram': {
                'zone': 'Coastal Plain',
                'soil_types': ['Alluvial', 'Red'],
                'common_crops': ['Rice', 'Vegetables', 'Groundnut']
            },
            'Pallavaram': {
                'zone': 'Coastal Plain',
                'soil_types': ['Alluvial', 'Red'],
                'common_crops': ['Rice', 'Vegetables']
            },
            
            # Western Zone
            'Coimbatore': {
                'zone': 'Western Zone',
                'soil_types': ['Red', 'Black'],
                'common_crops': ['Cotton', 'Maize', 'Turmeric', 'Coconut']
            },
            'Pollachi': {
                'zone': 'Western Zone',
                'soil_types': ['Red', 'Black'],
                'common_crops': ['Cotton', 'Coconut', 'Vegetables']
            },
            'Mettupalayam': {
                'zone': 'Western Zone',
                'soil_types': ['Red'],
                'common_crops': ['Tea', 'Coffee', 'Vegetables']
            },
            'Valparai': {
                'zone': 'Hilly Zone',
                'soil_types': ['Red', 'Clay'],
                'common_crops': ['Tea', 'Coffee', 'Cardamom']
            },
            'Erode': {
                'zone': 'Western Zone',
                'soil_types': ['Red', 'Black'],
                'common_crops': ['Turmeric', 'Cotton', 'Sugarcane', 'Maize']
            },
            'Gobichettipalayam': {
                'zone': 'Western Zone',
                'soil_types': ['Red', 'Black'],
                'common_crops': ['Turmeric', 'Cotton', 'Groundnut']
            },
            'Bhavani': {
                'zone': 'Western Zone',
                'soil_types': ['Red', 'Alluvial'],
                'common_crops': ['Turmeric', 'Cotton', 'Sugarcane']
            },
            'Perundurai': {
                'zone': 'Western Zone',
                'soil_types': ['Red'],
                'common_crops': ['Cotton', 'Turmeric', 'Groundnut']
            },
            'Tiruppur': {
                'zone': 'Western Zone',
                'soil_types': ['Red'],
                'common_crops': ['Cotton', 'Groundnut', 'Coconut']
            },
            'Nilgiris': {
                'zone': 'Hilly Zone',
                'soil_types': ['Red', 'Clay'],
                'common_crops': ['Tea', 'Coffee', 'Vegetables']
            },
            
            # Southern Zone
            'Madurai': {
                'zone': 'Southern Zone',
                'soil_types': ['Red', 'Black'],
                'common_crops': ['Cotton', 'Millets', 'Pulses', 'Groundnut']
            },
            'Melur': {
                'zone': 'Southern Zone',
                'soil_types': ['Red', 'Black'],
                'common_crops': ['Cotton', 'Groundnut', 'Millets']
            },
            'Usilampatti': {
                'zone': 'Southern Zone',
                'soil_types': ['Red'],
                'common_crops': ['Cotton', 'Millets', 'Pulses']
            },
            'Vadipatti': {
                'zone': 'Southern Zone',
                'soil_types': ['Red', 'Black'],
                'common_crops': ['Cotton', 'Groundnut', 'Vegetables']
            },
            'Dindigul': {
                'zone': 'Southern Zone',
                'soil_types': ['Red', 'Black'],
                'common_crops': ['Cotton', 'Millets', 'Vegetables']
            },
            'Theni': {
                'zone': 'Southern Zone',
                'soil_types': ['Red'],
                'common_crops': ['Cotton', 'Grapes', 'Cardamom']
            },
            'Virudhunagar': {
                'zone': 'Southern Zone',
                'soil_types': ['Red', 'Black'],
                'common_crops': ['Cotton', 'Groundnut', 'Millets']
            },
            'Ramanathapuram': {
                'zone': 'Southern Coastal',
                'soil_types': ['Red', 'Alluvial'],
                'common_crops': ['Cotton', 'Groundnut', 'Pulses']
            },
            'Sivaganga': {
                'zone': 'Southern Zone',
                'soil_types': ['Red', 'Black'],
                'common_crops': ['Cotton', 'Groundnut', 'Pulses']
            },
            
            # Cauvery Delta
            'Tiruchirappalli': {
                'zone': 'Cauvery Delta',
                'soil_types': ['Alluvial', 'Clay'],
                'common_crops': ['Rice', 'Sugarcane', 'Banana', 'Cotton']
            },
            'Trichy': {
                'zone': 'Cauvery Delta',
                'soil_types': ['Alluvial', 'Clay'],
                'common_crops': ['Rice', 'Sugarcane', 'Banana']
            },
            'Thottiyam': {
                'zone': 'Cauvery Delta',
                'soil_types': ['Alluvial', 'Red', 'Clay'],
                'common_crops': ['Rice', 'Sugarcane', 'Cotton', 'Groundnut']
            },
            'Lalgudi': {
                'zone': 'Cauvery Delta',
                'soil_types': ['Alluvial', 'Clay'],
                'common_crops': ['Rice', 'Sugarcane', 'Banana']
            },
            'Musiri': {
                'zone': 'Cauvery Delta',
                'soil_types': ['Alluvial', 'Red'],
                'common_crops': ['Rice', 'Groundnut', 'Cotton']
            },
            'Srirangam': {
                'zone': 'Cauvery Delta',
                'soil_types': ['Alluvial', 'Clay'],
                'common_crops': ['Rice', 'Sugarcane', 'Banana']
            },
            'Manachanallur': {
                'zone': 'Cauvery Delta',
                'soil_types': ['Alluvial', 'Red'],
                'common_crops': ['Rice', 'Groundnut', 'Cotton']
            },
            'Manapparai': {
                'zone': 'Cauvery Delta',
                'soil_types': ['Red', 'Black'],
                'common_crops': ['Cotton', 'Groundnut', 'Millets']
            },
            'Thuvarankurichi': {
                'zone': 'Cauvery Delta',
                'soil_types': ['Alluvial', 'Red'],
                'common_crops': ['Rice', 'Cotton', 'Groundnut']
            },
            'Thanjavur': {
                'zone': 'Cauvery Delta',
                'soil_types': ['Alluvial', 'Clay'],
                'common_crops': ['Rice', 'Sugarcane', 'Groundnut']
            },
            'Kumbakonam': {
                'zone': 'Cauvery Delta',
                'soil_types': ['Alluvial', 'Clay'],
                'common_crops': ['Rice', 'Sugarcane', 'Groundnut']
            },
            'Pattukkottai': {
                'zone': 'Cauvery Delta',
                'soil_types': ['Alluvial', 'Clay'],
                'common_crops': ['Rice', 'Groundnut', 'Cotton']
            },
            'Thiruvidaimarudur': {
                'zone': 'Cauvery Delta',
                'soil_types': ['Alluvial', 'Clay'],
                'common_crops': ['Rice', 'Sugarcane', 'Banana']
            },
            'Tiruvarur': {
                'zone': 'Cauvery Delta',
                'soil_types': ['Alluvial', 'Clay'],
                'common_crops': ['Rice', 'Sugarcane', 'Pulses']
            },
            'Nagapattinam': {
                'zone': 'Coastal Cauvery Delta',
                'soil_types': ['Alluvial', 'Clay'],
                'common_crops': ['Rice', 'Groundnut', 'Coconut']
            },
            'Pudukkottai': {
                'zone': 'Cauvery Delta',
                'soil_types': ['Red', 'Alluvial'],
                'common_crops': ['Rice', 'Cotton', 'Groundnut']
            },
            'Ariyalur': {
                'zone': 'Cauvery Delta',
                'soil_types': ['Alluvial', 'Red'],
                'common_crops': ['Rice', 'Groundnut', 'Cotton']
            },
            
            # Northwestern Zone
            'Salem': {
                'zone': 'Northwestern Zone',
                'soil_types': ['Red', 'Black'],
                'common_crops': ['Cotton', 'Groundnut', 'Maize', 'Turmeric']
            },
            'Attur': {
                'zone': 'Northwestern Zone',
                'soil_types': ['Red'],
                'common_crops': ['Groundnut', 'Cotton', 'Millets']
            },
            'Mettur': {
                'zone': 'Northwestern Zone',
                'soil_types': ['Red', 'Alluvial'],
                'common_crops': ['Sugarcane', 'Cotton', 'Maize']
            },
            'Omalur': {
                'zone': 'Northwestern Zone',
                'soil_types': ['Red'],
                'common_crops': ['Groundnut', 'Millets', 'Vegetables']
            },
            'Namakkal': {
                'zone': 'Northwestern Zone',
                'soil_types': ['Red'],
                'common_crops': ['Cotton', 'Maize', 'Groundnut']
            },
            'Mohanur': {
                'zone': 'Northwestern Zone',
                'soil_types': ['Red', 'Black'],
                'common_crops': ['Cotton', 'Groundnut', 'Maize', 'Pulses']
            },
            'Rasipuram': {
                'zone': 'Northwestern Zone',
                'soil_types': ['Red', 'Black'],
                'common_crops': ['Cotton', 'Maize', 'Groundnut']
            },
            'Tiruchengode': {
                'zone': 'Northwestern Zone',
                'soil_types': ['Red'],
                'common_crops': ['Cotton', 'Groundnut', 'Turmeric']
            },
            'Paramathi Velur': {
                'zone': 'Northwestern Zone',
                'soil_types': ['Red'],
                'common_crops': ['Millets', 'Groundnut', 'Cotton']
            },
            'Kolli Hills': {
                'zone': 'Hilly Zone',
                'soil_types': ['Red', 'Clay'],
                'common_crops': ['Coffee', 'Pepper', 'Fruits', 'Vegetables']
            },
            'Dharmapuri': {
                'zone': 'Northwestern Zone',
                'soil_types': ['Red'],
                'common_crops': ['Millets', 'Groundnut', 'Pulses']
            },
            'Krishnagiri': {
                'zone': 'Northwestern Zone',
                'soil_types': ['Red'],
                'common_crops': ['Millets', 'Groundnut', 'Mango']
            },
            
            # Northern Zone
            'Vellore': {
                'zone': 'Northern Zone',
                'soil_types': ['Red', 'Alluvial'],
                'common_crops': ['Rice', 'Groundnut', 'Sugarcane']
            },
            'Tiruvannamalai': {
                'zone': 'Northern Zone',
                'soil_types': ['Red'],
                'common_crops': ['Rice', 'Groundnut', 'Sugarcane']
            },
            'Ranipet': {
                'zone': 'Northern Zone',
                'soil_types': ['Red', 'Alluvial'],
                'common_crops': ['Rice', 'Groundnut', 'Vegetables']
            },
            'Tirupattur': {
                'zone': 'Northern Zone',
                'soil_types': ['Red'],
                'common_crops': ['Millets', 'Groundnut', 'Mango']
            },
            
            # Coastal Zone
            'Cuddalore': {
                'zone': 'Coastal Zone',
                'soil_types': ['Alluvial', 'Clay'],
                'common_crops': ['Rice', 'Groundnut', 'Cashew']
            },
            'Villupuram': {
                'zone': 'Coastal Zone',
                'soil_types': ['Alluvial', 'Red'],
                'common_crops': ['Rice', 'Groundnut', 'Sugarcane']
            },
            'Kallakurichi': {
                'zone': 'Coastal Zone',
                'soil_types': ['Alluvial', 'Red'],
                'common_crops': ['Rice', 'Sugarcane', 'Groundnut']
            },
            
            # Southern Coastal
            'Tirunelveli': {
                'zone': 'Southern Coastal Zone',
                'soil_types': ['Red', 'Alluvial'],
                'common_crops': ['Rice', 'Cotton', 'Groundnut']
            },
            'Thoothukudi': {
                'zone': 'Southern Coastal Zone',
                'soil_types': ['Red', 'Alluvial'],
                'common_crops': ['Cotton', 'Groundnut', 'Pulses']
            },
            'Tenkasi': {
                'zone': 'Southern Coastal Zone',
                'soil_types': ['Red'],
                'common_crops': ['Rice', 'Banana', 'Vegetables']
            },
            
            # Other Districts
            'Kanchipuram': {
                'zone': 'Northern Coastal Zone',
                'soil_types': ['Alluvial', 'Red'],
                'common_crops': ['Rice', 'Groundnut', 'Vegetables']
            },
            'Chengalpattu': {
                'zone': 'Northern Coastal Zone',
                'soil_types': ['Alluvial', 'Red'],
                'common_crops': ['Rice', 'Vegetables', 'Groundnut']
            },
            'Karur': {
                'zone': 'Central Zone',
                'soil_types': ['Red', 'Alluvial'],
                'common_crops': ['Cotton', 'Maize', 'Groundnut']
            },
            'Perambalur': {
                'zone': 'Central Zone',
                'soil_types': ['Red', 'Alluvial'],
                'common_crops': ['Rice', 'Sugarcane', 'Cotton']
            },
            'Kanniyakumari': {
                'zone': 'Southern Tip',
                'soil_types': ['Red', 'Alluvial'],
                'common_crops': ['Rubber', 'Rice', 'Coconut', 'Banana']
            }
        }
        
        return tamil_nadu_zones.get(city, {
            'zone': 'General Agricultural Zone',
            'soil_types': ['Alluvial', 'Red', 'Black', 'Clay'],
            'common_crops': ['Rice', 'Cotton', 'Wheat', 'Millets']
        })


# Test the service
if __name__ == '__main__':
    geo_service = GeolocationService()
    
    print("Testing Geolocation Service...")
    print("-" * 50)
    
    # Test automatic location detection
    location = geo_service.get_current_location()
    print("\n1. Auto-detected Location:")
    print(f"   City: {location['city']}")
    print(f"   Region: {location['region']}")
    print(f"   Country: {location['country']}")
    print(f"   Coordinates: ({location['latitude']}, {location['longitude']})")
    print(f"   Source: {location['source']}")
    
    # Test with specific coordinates (Chennai)
    print("\n2. Reverse Geocoding (Chennai coordinates):")
    chennai_loc = geo_service.get_location_by_coords(13.0827, 80.2707)
    print(f"   City: {chennai_loc['city']}")
    print(f"   Region: {chennai_loc['region']}")
    
    # Test comprehensive location info
    print("\n3. Comprehensive Location Info:")
    full_info = geo_service.get_location_info()
    print(f"   Agricultural Zone: {full_info.get('agricultural_zone', {}).get('zone', 'N/A')}")
    print(f"   Common Crops: {', '.join(full_info.get('agricultural_zone', {}).get('common_crops', []))}")
