# main.py - main for neural network class - working now
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pandas as pd
import numpy as np
from betting_nn_wrapper import BettingNeuralWrapper

class SimpleBettingAlgo:
    def __init__(self):
        # Create sample data
        dates = pd.date_range(start='2022-01-01', periods=1000)
        
        self.historical_odds = pd.DataFrame({
            'match_id': range(1000),
            'match_date': dates,  # Make sure this column name matches
            'home_team': ['Team A'] * 1000,
            'away_team': ['Team B'] * 1000,
            'home_win_odds': np.random.uniform(1.5, 4.0, 1000),
            'draw_odds': np.random.uniform(2.0, 4.5, 1000),
            'away_win_odds': np.random.uniform(1.5, 4.0, 1000)
        })
        
        self.match_results = pd.DataFrame({
            'match_id': range(1000),
            'date': dates,  # Make sure this column name matches
            'home_team': ['Team A'] * 1000,
            'away_team': ['Team B'] * 1000,
            'winner': np.random.choice(['Team A', 'Team B', 'draw'], 1000)
        })
    
    def detect_market_inefficiency(self, match_id):
        # Simple placeholder implementation
        return {
            'edge': 0.1,
            'bet_type': 'home',
            'bookie': 'generic',
            'true_prob': 0.55,
            'implied_prob': 0.45
        }

def main():
    # Create instance of your betting algorithm
    algo = SimpleBettingAlgo()
    
    # Create neural wrapper
    nn_wrapper = BettingNeuralWrapper(algo)
    
    # Train the neural network
    print("Starting neural network training...")
    history = nn_wrapper.train_neural_network()
    
    # Enhance the predictions
    nn_wrapper.enhance_predictions()
    
    # Test a prediction
    test_match_id = 0
    enhanced_prediction = algo.detect_market_inefficiency(test_match_id)
    print("\nTest prediction for match_id 0:")
    print(enhanced_prediction)
    
    # Save the model
    print("\nSaving model...")
    nn_wrapper.save_model()

if __name__ == "__main__":
    main()
