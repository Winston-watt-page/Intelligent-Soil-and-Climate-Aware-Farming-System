"""
Soil Health Monitoring and Fertilizer Recommendations
Analyzes soil parameters and provides fertilizer suggestions
"""

class SoilHealthMonitor:
    """
    Monitors soil health and provides fertilizer recommendations based on:
    - NPK levels
    - pH levels
    - Organic matter content
    - Moisture levels
    - Crop requirements
    """
    
    def __init__(self):
        # Optimal ranges for soil parameters
        self.optimal_ranges = {
            'pH': {'min': 6.0, 'max': 7.5, 'ideal': 6.8},
            'nitrogen': {'min': 0.4, 'max': 0.8, 'ideal': 0.6},
            'phosphorus': {'min': 0.3, 'max': 0.7, 'ideal': 0.5},
            'potassium': {'min': 0.4, 'max': 0.8, 'ideal': 0.6},
            'organic_matter': {'min': 0.5, 'max': 1.0, 'ideal': 0.7},
            'moisture': {'min': 0.4, 'max': 0.8, 'ideal': 0.6}
        }
        
        # Fertilizer recommendations
        self.fertilizers = {
            'nitrogen_deficient': {
                'organic': ['Urea', 'Ammonium Sulfate', 'Compost', 'Chicken Manure'],
                'dosage': '40-60 kg/acre',
                'application': 'Apply in split doses during growth stages'
            },
            'phosphorus_deficient': {
                'organic': ['Single Super Phosphate', 'DAP', 'Bone Meal', 'Rock Phosphate'],
                'dosage': '30-50 kg/acre',
                'application': 'Apply at planting time'
            },
            'potassium_deficient': {
                'organic': ['Muriate of Potash', 'Wood Ash', 'Potassium Sulfate'],
                'dosage': '20-40 kg/acre',
                'application': 'Apply before flowering stage'
            },
            'acidic_soil': {
                'organic': ['Lime (Calcium Carbonate)', 'Dolomite', 'Wood Ash'],
                'dosage': '200-500 kg/acre',
                'application': 'Apply 2-3 months before planting'
            },
            'alkaline_soil': {
                'organic': ['Sulfur', 'Gypsum', 'Organic Compost', 'Peat Moss'],
                'dosage': '100-300 kg/acre',
                'application': 'Apply and mix thoroughly'
            },
            'low_organic_matter': {
                'organic': ['Farmyard Manure', 'Compost', 'Green Manure', 'Vermicompost'],
                'dosage': '5-10 tons/acre',
                'application': 'Apply before plowing'
            }
        }
    
    def analyze_soil_health(self, soil_features):
        """
        Analyze soil health parameters
        
        Args:
            soil_features: dict with pH, NPK, moisture, organic_matter
        
        Returns:
            dict with health status and issues
        """
        issues = []
        health_score = 100
        
        # Check pH
        pH = soil_features.get('pH', 7.0)
        if pH < 5.5:
            issues.append('Highly acidic soil')
            health_score -= 20
        elif pH < self.optimal_ranges['pH']['min']:
            issues.append('Slightly acidic soil')
            health_score -= 10
        elif pH > 8.5:
            issues.append('Highly alkaline soil')
            health_score -= 20
        elif pH > self.optimal_ranges['pH']['max']:
            issues.append('Slightly alkaline soil')
            health_score -= 10
        
        # Check Nitrogen
        nitrogen = soil_features.get('nitrogen', 0.5)
        if nitrogen < self.optimal_ranges['nitrogen']['min']:
            issues.append('Low nitrogen levels')
            health_score -= 15
        
        # Check Phosphorus
        phosphorus = soil_features.get('phosphorus', 0.5)
        if phosphorus < self.optimal_ranges['phosphorus']['min']:
            issues.append('Low phosphorus levels')
            health_score -= 15
        
        # Check Potassium
        potassium = soil_features.get('potassium', 0.5)
        if potassium < self.optimal_ranges['potassium']['min']:
            issues.append('Low potassium levels')
            health_score -= 15
        
        # Check Organic Matter
        organic_matter = soil_features.get('organic_matter', 0.5)
        if organic_matter < self.optimal_ranges['organic_matter']['min']:
            issues.append('Low organic matter')
            health_score -= 15
        
        # Check Moisture
        moisture = soil_features.get('moisture', 0.5)
        if moisture < 0.3:
            issues.append('Insufficient moisture')
            health_score -= 10
        elif moisture > 0.9:
            issues.append('Excess moisture - drainage needed')
            health_score -= 10
        
        # Determine health status
        if health_score >= 90:
            status = 'Excellent'
            status_color = '#27ae60'
        elif health_score >= 75:
            status = 'Good'
            status_color = '#2ecc71'
        elif health_score >= 60:
            status = 'Fair'
            status_color = '#f39c12'
        elif health_score >= 40:
            status = 'Poor'
            status_color = '#e67e22'
        else:
            status = 'Critical'
            status_color = '#e74c3c'
        
        return {
            'health_score': max(0, health_score),
            'status': status,
            'status_color': status_color,
            'issues': issues,
            'parameters': {
                'pH': {'value': pH, 'optimal': self.optimal_ranges['pH']['ideal'], 'status': self._get_param_status(pH, 'pH')},
                'nitrogen': {'value': nitrogen, 'optimal': self.optimal_ranges['nitrogen']['ideal'], 'status': self._get_param_status(nitrogen, 'nitrogen')},
                'phosphorus': {'value': phosphorus, 'optimal': self.optimal_ranges['phosphorus']['ideal'], 'status': self._get_param_status(phosphorus, 'phosphorus')},
                'potassium': {'value': potassium, 'optimal': self.optimal_ranges['potassium']['ideal'], 'status': self._get_param_status(potassium, 'potassium')},
                'organic_matter': {'value': organic_matter, 'optimal': self.optimal_ranges['organic_matter']['ideal'], 'status': self._get_param_status(organic_matter, 'organic_matter')},
                'moisture': {'value': moisture, 'optimal': self.optimal_ranges['moisture']['ideal'], 'status': self._get_param_status(moisture, 'moisture')}
            }
        }
    
    def _get_param_status(self, value, param_name):
        """Get status for a parameter"""
        optimal = self.optimal_ranges[param_name]
        if optimal['min'] <= value <= optimal['max']:
            return 'Optimal'
        elif value < optimal['min']:
            return 'Low'
        else:
            return 'High'
    
    def get_fertilizer_recommendations(self, soil_features, crop_name=None):
        """
        Get fertilizer recommendations based on soil analysis
        
        Args:
            soil_features: dict with soil parameters
            crop_name: optional crop name for specific recommendations
        
        Returns:
            list of fertilizer recommendations
        """
        recommendations = []
        
        pH = soil_features.get('pH', 7.0)
        nitrogen = soil_features.get('nitrogen', 0.5)
        phosphorus = soil_features.get('phosphorus', 0.5)
        potassium = soil_features.get('potassium', 0.5)
        organic_matter = soil_features.get('organic_matter', 0.5)
        
        # pH corrections
        if pH < 6.0:
            rec = self.fertilizers['acidic_soil'].copy()
            rec['issue'] = 'Acidic Soil Correction'
            rec['priority'] = 'High'
            recommendations.append(rec)
        elif pH > 7.5:
            rec = self.fertilizers['alkaline_soil'].copy()
            rec['issue'] = 'Alkaline Soil Correction'
            rec['priority'] = 'High'
            recommendations.append(rec)
        
        # NPK deficiencies
        if nitrogen < self.optimal_ranges['nitrogen']['min']:
            rec = self.fertilizers['nitrogen_deficient'].copy()
            rec['issue'] = 'Nitrogen Deficiency'
            rec['priority'] = 'High' if nitrogen < 0.3 else 'Medium'
            recommendations.append(rec)
        
        if phosphorus < self.optimal_ranges['phosphorus']['min']:
            rec = self.fertilizers['phosphorus_deficient'].copy()
            rec['issue'] = 'Phosphorus Deficiency'
            rec['priority'] = 'High' if phosphorus < 0.2 else 'Medium'
            recommendations.append(rec)
        
        if potassium < self.optimal_ranges['potassium']['min']:
            rec = self.fertilizers['potassium_deficient'].copy()
            rec['issue'] = 'Potassium Deficiency'
            rec['priority'] = 'High' if potassium < 0.3 else 'Medium'
            recommendations.append(rec)
        
        # Organic matter
        if organic_matter < self.optimal_ranges['organic_matter']['min']:
            rec = self.fertilizers['low_organic_matter'].copy()
            rec['issue'] = 'Low Organic Matter'
            rec['priority'] = 'Medium'
            recommendations.append(rec)
        
        # General maintenance if soil is healthy
        if not recommendations:
            recommendations.append({
                'issue': 'Maintenance',
                'priority': 'Low',
                'organic': ['Balanced NPK (10-10-10)', 'Compost', 'Vermicompost'],
                'dosage': '20-30 kg/acre',
                'application': 'Apply before planting and at flowering stage'
            })
        
        return recommendations
    
    def get_improvement_plan(self, soil_features, timeframe_weeks=12):
        """
        Get a step-by-step soil improvement plan
        
        Args:
            soil_features: dict with soil parameters
            timeframe_weeks: improvement timeframe in weeks
        
        Returns:
            list of timed actions
        """
        plan = []
        health_analysis = self.analyze_soil_health(soil_features)
        
        # Week 1-2: Testing and pH correction
        plan.append({
            'week': '1-2',
            'action': 'Soil Testing & pH Correction',
            'tasks': [
                'Conduct comprehensive soil test',
                'Apply pH correctors if needed (lime/sulfur)',
                'Test drainage and improve if necessary'
            ]
        })
        
        # Week 3-4: Organic matter addition
        if soil_features.get('organic_matter', 0.5) < 0.5:
            plan.append({
                'week': '3-4',
                'action': 'Organic Matter Enhancement',
                'tasks': [
                    'Add compost or farmyard manure (5-10 tons/acre)',
                    'Incorporate green manure if available',
                    'Till to mix organic matter thoroughly'
                ]
            })
        
        # Week 5-6: NPK application
        plan.append({
            'week': '5-6',
            'action': 'Nutrient Balancing',
            'tasks': [
                'Apply nitrogen fertilizer (40-60 kg/acre)',
                'Apply phosphorus at planting depth',
                'Add potassium if deficient'
            ]
        })
        
        # Week 7-8: Micronutrients and monitoring
        plan.append({
            'week': '7-8',
            'action': 'Micronutrient Addition & Monitoring',
            'tasks': [
                'Apply micronutrient mix if needed',
                'Monitor soil moisture regularly',
                'Check for pest/disease signs'
            ]
        })
        
        # Week 9-12: Maintenance and planting preparation
        plan.append({
            'week': '9-12',
            'action': 'Final Preparation',
            'tasks': [
                'Conduct follow-up soil test',
                'Apply final dose of fertilizers',
                'Prepare field for planting',
                'Ensure proper irrigation setup'
            ]
        })
        
        return plan


# Test the service
if __name__ == '__main__':
    monitor = SoilHealthMonitor()
    
    print("Testing Soil Health Monitor...")
    print("-" * 50)
    
    # Test with sample soil data
    sample_soil = {
        'pH': 5.8,
        'nitrogen': 0.3,
        'phosphorus': 0.4,
        'potassium': 0.5,
        'organic_matter': 0.4,
        'moisture': 0.6
    }
    
    # Test health analysis
    print("\n1. Soil Health Analysis:")
    health = monitor.analyze_soil_health(sample_soil)
    print(f"   Health Score: {health['health_score']}/100")
    print(f"   Status: {health['status']}")
    print(f"   Issues: {', '.join(health['issues']) if health['issues'] else 'None'}")
    
    # Test fertilizer recommendations
    print("\n2. Fertilizer Recommendations:")
    fertilizers = monitor.get_fertilizer_recommendations(sample_soil)
    for i, rec in enumerate(fertilizers, 1):
        print(f"   {i}. {rec['issue']} ({rec['priority']} priority)")
        print(f"      Options: {', '.join(rec['organic'][:2])}")
        print(f"      Dosage: {rec['dosage']}")
    
    # Test improvement plan
    print("\n3. 12-Week Improvement Plan:")
    plan = monitor.get_improvement_plan(sample_soil)
    for step in plan:
        print(f"   Week {step['week']}: {step['action']}")
        for task in step['tasks'][:2]:
            print(f"      - {task}")
