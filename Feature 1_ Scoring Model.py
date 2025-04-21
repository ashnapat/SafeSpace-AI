import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import logging
from math import radians, sin, cos, sqrt, atan2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SiteScorer:
    """
    A class to evaluate and score potential Emergency Interim Housing (EIH) sites
    based on multiple criteria including proximity to services, infrastructure,
    and community impact.
    """
    
    def __init__(self, city_boundary: pd.DataFrame):
        """
        Initialize the SiteScorer.
        
        Args:
            city_boundary (pd.DataFrame): DataFrame containing San Jose city data
        """
        self.city_boundary = city_boundary
        
        # Define scoring weights
        self.weights = {
            'services': {
                'public_transit': 0.10,
                'healthcare': 0.10,
                'grocery': 0.10,
                'social_services': 0.10
            },
            'infrastructure': {
                'utilities': 0.10,
                'road_connectivity': 0.10,
                'emergency_response': 0.10
            },
            'community': {
                'population_density': 0.10,
                'demographic_risk': 0.10,
                'environmental_justice': 0.10
            }
        }

    def haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate the great circle distance between two points on Earth.
        
        Args:
            lat1, lon1: Latitude and longitude of first point
            lat2, lon2: Latitude and longitude of second point
            
        Returns:
            Distance in kilometers
        """
        R = 6371  # Earth's radius in kilometers

        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        distance = R * c

        return distance

    def simulate_service_proximity(self, location: Tuple[float, float]) -> float:
        """
        Simulate service proximity score based on location.
        In a real implementation, this would query actual service locations.
        
        Args:
            location: Tuple of (latitude, longitude)
            
        Returns:
            float: Score between 0 and 1
        """
        # Define key service locations in San Jose
        key_locations = {
            'downtown': (37.3382, -121.8863),  # Downtown San Jose
            'diridon': (37.3297, -121.9018),   # Diridon Station
            'valley_med': (37.3166, -121.9277), # Valley Medical Center
            'eastridge': (37.3254, -121.8157)   # Eastridge Mall
        }
        
        # Calculate minimum distance to any key location
        distances = [
            self.haversine_distance(location[0], location[1], lat, lon)
            for lat, lon in key_locations.values()
        ]
        min_distance = min(distances)
        
        # Score decreases with distance, capped at 5km
        return max(0, 1 - (min_distance / 5))

    def simulate_infrastructure_score(self, location: Tuple[float, float]) -> float:
        """
        Simulate infrastructure availability score.
        In a real implementation, this would check actual infrastructure data.
        
        Args:
            location: Tuple of (latitude, longitude)
            
        Returns:
            float: Score between 0 and 1
        """
        # Define infrastructure hubs
        hubs = {
            'downtown': (37.3382, -121.8863),
            'north': (37.4034, -121.8863),
            'south': (37.2788, -121.8863),
            'east': (37.3382, -121.8163),
            'west': (37.3382, -121.9563)
        }
        
        # Calculate distances to infrastructure hubs
        distances = [
            self.haversine_distance(location[0], location[1], lat, lon)
            for lat, lon in hubs.values()
        ]
        min_distance = min(distances)
        
        # Score decreases with distance from nearest hub, capped at 8km
        return max(0, 1 - (min_distance / 8))

    def calculate_community_impact_score(self, location: Tuple[float, float], 
                                      demographic_data: pd.DataFrame) -> float:
        """
        Calculate community impact score based on demographic factors.
        
        Args:
            location: Tuple of (latitude, longitude)
            demographic_data: DataFrame with demographic information
            
        Returns:
            float: Score between 0 and 1
        """
        try:
            # For demo purposes, use random selection from demographic data
            # In a real implementation, this would use actual census tract data
            tract_data = demographic_data.sample(n=1).iloc[0]
            
            # Calculate sub-scores
            population_density_score = 1 - min(1.0, tract_data['population_density'] / 10000)
            poverty_rate_score = min(1.0, tract_data['poverty_rate'] / 30)
            environmental_score = 1 - min(1.0, tract_data['calenviroscreen_score'] / 100)
            
            # Weight the sub-scores
            return (population_density_score * 0.3 + 
                   poverty_rate_score * 0.4 + 
                   environmental_score * 0.3)
            
        except Exception as e:
            logger.error(f"Error calculating community impact score: {str(e)}")
            return 0.0

    def score_location(self, location: Tuple[float, float], 
                      demographic_data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate overall suitability score for a potential EIH site.
        
        Args:
            location: Tuple of (latitude, longitude)
            demographic_data: DataFrame with demographic information
            
        Returns:
            Dict containing overall score and component scores
        """
        # Calculate service proximity scores
        transit_score = self.simulate_service_proximity(location)
        healthcare_score = self.simulate_service_proximity(location)
        grocery_score = self.simulate_service_proximity(location)
        social_services_score = self.simulate_service_proximity(location)
        
        # Calculate infrastructure score
        infrastructure_score = self.simulate_infrastructure_score(location)
        
        # Calculate community impact score
        community_score = self.calculate_community_impact_score(location, demographic_data)
        
        # Calculate weighted total score
        total_score = (
            transit_score * self.weights['services']['public_transit'] +
            healthcare_score * self.weights['services']['healthcare'] +
            grocery_score * self.weights['services']['grocery'] +
            social_services_score * self.weights['services']['social_services'] +
            infrastructure_score * (
                self.weights['infrastructure']['utilities'] +
                self.weights['infrastructure']['road_connectivity'] +
                self.weights['infrastructure']['emergency_response']
            ) / 3 +
            community_score * (
                self.weights['community']['population_density'] +
                self.weights['community']['demographic_risk'] +
                self.weights['community']['environmental_justice']
            ) / 3
        )
        
        return {
            'total_score': total_score,
            'component_scores': {
                'transit': transit_score,
                'healthcare': healthcare_score,
                'grocery': grocery_score,
                'social_services': social_services_score,
                'infrastructure': infrastructure_score,
                'community_impact': community_score
            }
        }

    def get_top_locations(self, candidate_locations: List[Tuple[float, float]], 
                         demographic_data: pd.DataFrame, 
                         n: int = 5) -> pd.DataFrame:
        """
        Score multiple locations and return the top N candidates.
        
        Args:
            candidate_locations: List of (latitude, longitude) tuples
            demographic_data: DataFrame with demographic information
            n: Number of top locations to return
            
        Returns:
            DataFrame with scored locations
        """
        results = []
        
        for location in candidate_locations:
            scores = self.score_location(location, demographic_data)
            results.append({
                'latitude': location[0],
                'longitude': location[1],
                'total_score': scores['total_score'],
                **scores['component_scores']
            })
        
        results_df = pd.DataFrame(results)
        return results_df.nlargest(n, 'total_score') 