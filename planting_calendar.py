"""
Optimal Planting Calendar
Provides planting recommendations based on historical data, weather patterns, and soil type
"""

from datetime import datetime, timedelta
import calendar


class PlantingCalendar:
    """
    Recommends optimal planting times based on:
    - Soil type
    - Crop requirements
    - Historical weather patterns
    - Regional agricultural calendar
    """
    
    def __init__(self):
        # Optimal planting months for each crop by soil type (Tamil Nadu region)
        self.planting_windows = {
            0: {  # Alluvial Soil
                'Rice': {'months': [6, 7, 11, 12], 'season': 'Kharif & Rabi', 'duration_days': 120},
                'Wheat': {'months': [11, 12], 'season': 'Rabi', 'duration_days': 120},
                'Sugarcane': {'months': [1, 2, 10, 11], 'season': 'Year-round', 'duration_days': 365},
                'Maize': {'months': [6, 7, 11], 'season': 'Kharif & Rabi', 'duration_days': 90},
                'Cotton': {'months': [6, 7, 8], 'season': 'Kharif', 'duration_days': 180},
                'Soybean': {'months': [6, 7], 'season': 'Kharif', 'duration_days': 100},
                'Jute': {'months': [3, 4], 'season': 'Summer', 'duration_days': 120},
                'Vegetables': {'months': [1, 2, 6, 7, 10], 'season': 'Multiple', 'duration_days': 60}
            },
            1: {  # Black Soil
                'Cotton': {'months': [6, 7, 8], 'season': 'Kharif', 'duration_days': 180},
                'Wheat': {'months': [11, 12], 'season': 'Rabi', 'duration_days': 120},
                'Jowar': {'months': [6, 7, 10, 11], 'season': 'Kharif & Rabi', 'duration_days': 100},
                'Millets': {'months': [6, 7], 'season': 'Kharif', 'duration_days': 80},
                'Linseed': {'months': [10, 11], 'season': 'Rabi', 'duration_days': 120},
                'Castor': {'months': [6, 7], 'season': 'Kharif', 'duration_days': 150},
                'Sunflower': {'months': [6, 7, 11], 'season': 'Kharif & Rabi', 'duration_days': 90},
                'Groundnut': {'months': [6, 7], 'season': 'Kharif', 'duration_days': 120}
            },
            2: {  # Clay Soil
                'Rice': {'months': [6, 7, 11, 12], 'season': 'Kharif & Rabi', 'duration_days': 120},
                'Lettuce': {'months': [9, 10, 11], 'season': 'Winter', 'duration_days': 60},
                'Cabbage': {'months': [9, 10, 11], 'season': 'Winter', 'duration_days': 90},
                'Broccoli': {'months': [9, 10], 'season': 'Winter', 'duration_days': 85},
                'Chard': {'months': [8, 9, 10], 'season': 'Fall-Winter', 'duration_days': 55},
                'Snap Beans': {'months': [2, 3, 9, 10], 'season': 'Spring & Fall', 'duration_days': 60},
                'Cauliflower': {'months': [9, 10], 'season': 'Winter', 'duration_days': 100},
                'Spinach': {'months': [9, 10, 11], 'season': 'Winter', 'duration_days': 45}
            },
            3: {  # Red Soil
                'Cotton': {'months': [6, 7, 8], 'season': 'Kharif', 'duration_days': 180},
                'Millets': {'months': [6, 7], 'season': 'Kharif', 'duration_days': 80},
                'Pulses': {'months': [10, 11], 'season': 'Rabi', 'duration_days': 90},
                'Wheat': {'months': [11, 12], 'season': 'Rabi', 'duration_days': 120},
                'Groundnut': {'months': [6, 7], 'season': 'Kharif', 'duration_days': 120},
                'Potatoes': {'months': [10, 11, 12], 'season': 'Rabi', 'duration_days': 90},
                'Oil Seeds': {'months': [6, 7, 11], 'season': 'Kharif & Rabi', 'duration_days': 100},
                'Tobacco': {'months': [6, 7], 'season': 'Kharif', 'duration_days': 120}
            }
        }
        
        # Month names
        self.month_names = [
            '', 'January', 'February', 'March', 'April', 'May', 'June',
            'July', 'August', 'September', 'October', 'November', 'December'
        ]
        
        self.month_names_tamil = [
            '', 'ஜனவரி', 'பிப்ரவரி', 'மார்ச்', 'ஏப்ரல்', 'மே', 'ஜூன்',
            'ஜூலை', 'ஆகஸ்ட்', 'செப்டம்பர்', 'அக்டோபர்', 'நவம்பர்', 'டிசம்பர்'
        ]
    
    def get_planting_schedule(self, soil_type, crop_name):
        """
        Get optimal planting schedule for a crop
        
        Args:
            soil_type: int (0-3)
            crop_name: str
        
        Returns:
            dict with planting schedule information
        """
        if crop_name not in self.planting_windows[soil_type]:
            return None
        
        crop_data = self.planting_windows[soil_type][crop_name]
        current_month = datetime.now().month
        
        # Find next optimal planting month
        next_planting = None
        for month in crop_data['months']:
            if month >= current_month:
                next_planting = month
                break
        
        if next_planting is None:
            next_planting = crop_data['months'][0]  # First month next year
        
        # Calculate harvest date
        planting_date = datetime(datetime.now().year if next_planting >= current_month else datetime.now().year + 1,
                                next_planting, 15)
        harvest_date = planting_date + timedelta(days=crop_data['duration_days'])
        
        return {
            'crop': crop_name,
            'optimal_months': [self.month_names[m] for m in crop_data['months']],
            'optimal_months_tamil': [self.month_names_tamil[m] for m in crop_data['months']],
            'season': crop_data['season'],
            'next_planting_month': self.month_names[next_planting],
            'next_planting_date': planting_date.strftime('%B %d, %Y'),
            'expected_harvest': harvest_date.strftime('%B %d, %Y'),
            'duration_days': crop_data['duration_days'],
            'is_current_window': current_month in crop_data['months']
        }
    
    def get_current_recommendations(self, soil_type):
        """
        Get crops that can be planted in current month
        
        Args:
            soil_type: int (0-3)
        
        Returns:
            list of crops suitable for current month
        """
        current_month = datetime.now().month
        recommendations = []
        
        for crop_name, crop_data in self.planting_windows[soil_type].items():
            if current_month in crop_data['months']:
                recommendations.append({
                    'crop': crop_name,
                    'season': crop_data['season'],
                    'duration_days': crop_data['duration_days'],
                    'urgency': 'Plant now!' if current_month == crop_data['months'][-1] else 'Good time to plant'
                })
        
        return recommendations
    
    def get_full_calendar(self, soil_type, crops):
        """
        Get complete planting calendar for multiple crops
        
        Args:
            soil_type: int (0-3)
            crops: list of crop names
        
        Returns:
            dict with month-by-month planting guide
        """
        calendar_data = {}
        
        for crop in crops:
            schedule = self.get_planting_schedule(soil_type, crop)
            if schedule:
                calendar_data[crop] = schedule
        
        return calendar_data


# Test the service
if __name__ == '__main__':
    planting_cal = PlantingCalendar()
    
    print("Testing Planting Calendar Service...")
    print("-" * 50)
    
    # Test for Alluvial soil
    print("\n1. Rice Planting Schedule (Alluvial Soil):")
    schedule = planting_cal.get_planting_schedule(0, 'Rice')
    print(f"   Optimal Months: {', '.join(schedule['optimal_months'])}")
    print(f"   Season: {schedule['season']}")
    print(f"   Next Planting: {schedule['next_planting_date']}")
    print(f"   Expected Harvest: {schedule['expected_harvest']}")
    print(f"   Duration: {schedule['duration_days']} days")
    
    # Test current recommendations
    print("\n2. Crops to Plant Now (Alluvial Soil):")
    current = planting_cal.get_current_recommendations(0)
    for rec in current:
        print(f"   - {rec['crop']}: {rec['urgency']} ({rec['season']})")
    
    # Test full calendar
    print("\n3. Full Calendar for Top 3 Crops:")
    calendar = planting_cal.get_full_calendar(0, ['Rice', 'Wheat', 'Sugarcane'])
    for crop, data in calendar.items():
        print(f"   {crop}: Plant in {', '.join(data['optimal_months'])}")
