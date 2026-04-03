"""
Geolocation Service
Automatically detects user's location based on IP address
"""

import requests
import json
import os


class GeolocationService:
    """
    Provides automatic geolocation detection using IP-based services.
    Uses OpenStreetMap (Nominatim) for free, high-quality reverse geocoding.
    """
    
    def __init__(self):
        # --- API Endpoints ---
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
        print("WARNING: IP geolocation is disabled to avoid inaccurate location results.")
        print("TIP: Use browser geolocation or select a city manually for accurate results.")
        return {
            'error': 'IP-based auto-detection is disabled because it is inaccurate. Use GPS or select a city manually.'
        }
    
    def get_location_by_coords(self, lat, lon):
        """
        Reverse geocoding: Get location details from coordinates.
        Uses OpenStreetMap (Nominatim) as the primary provider for its accuracy
        with village-level data in India and falls back to other services.
        
        Args:
            lat: Latitude
            lon: Longitude
        
        Returns:
            dict with location details
        """
        print(f"Reverse geocoding coordinates: {lat}, {lon}")

        # --- Method 1: Nominatim (OpenStreetMap) - Free & Excellent for Village Data ---
        # Attribution: Remember to credit "© OpenStreetMap contributors" in the UI.
        try:
            url = "https://nominatim.openstreetmap.org/reverse"
            params = {
                'lat': lat,
                'lon': lon,
                'format': 'json',
                'addressdetails': 1,
                'accept-language': 'en' # Ensure English results
            }
            headers = {
                'User-Agent': 'IntelligentFarmingSystem/1.0 (Educational Project)'
            }
            
            response = requests.get(url, params=params, headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                address = data.get('address', {})
                
                # Prioritize specific location types for accuracy in rural India
                village = address.get('village', '')
                hamlet = address.get('hamlet', '')
                suburb = address.get('suburb', '')
                town = address.get('town', '')
                city = address.get('city', '')
                
                # Find the most specific name available
                specific_location = village or hamlet or suburb or town or city or 'Unknown'
                
                district = address.get('county', address.get('state_district', 'Unknown'))
                state = address.get('state', 'Unknown')
                country = address.get('country', 'Unknown')
                country_code = address.get('country_code', 'un').upper()

                print(f"Nominatim Success: {specific_location}, {district}, {state}")
                
                return {
                    'latitude': float(lat),
                    'longitude': float(lon),
                    'city': specific_location,
                    'village': village,
                    'town': town,
                    'district': district,
                    'state': state,
                    'country': country,
                    'country_code': country_code,
                    'display_name': data.get('display_name', f"{specific_location}, {state}"),
                    'source': 'nominatim'
                }
        except Exception as e:
            print(f"WARNING: Nominatim (OpenStreetMap) failed: {e}")

        # --- Method 2: BigDataCloud (Fallback) ---
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
                
                city = data.get('city', '')
                locality = data.get('locality', '')
                principalSubdiv = data.get('principalSubdivision', 'Unknown')
                country = data.get('countryName', 'Unknown')
                country_code = data.get('countryCode', 'UN')
                
                specific_location = locality or city or principalSubdiv
                
                print(f"BigDataCloud Fallback: {specific_location}, {principalSubdiv}, {country}")
                
                return {
                    'latitude': float(lat),
                    'longitude': float(lon),
                    'city': specific_location,
                    'village': '', # Not provided by this API
                    'town': '', # Not provided by this API
                    'district': principalSubdiv,
                    'state': principalSubdiv,
                    'country': country,
                    'country_code': country_code,
                    'display_name': f"{specific_location}, {principalSubdiv}",
                    'source': 'bigdatacloud'
                }
        except Exception as e:
            print(f"WARNING: BigDataCloud failed: {e}")
        
        # --- Final Fallback: Return coordinates only ---
        print(f"WARNING: All geocoding services failed. Returning coordinates only.")
        return {
            'latitude': float(lat),
            'longitude': float(lon),
            'city': f'Location ({round(lat, 2)}, {round(lon, 2)})',
            'village': '',
            'town': '',
            'district': 'Unknown',
            'state': 'Unknown',
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
