#############################
#############################
#############################

# shit file to far - trying to get it to work

#############################
#############################
#############################

import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers, models
from sklearn.preprocessing import StandardScaler
import joblib
from datetime import datetime, timedelta

class BettingNeuralWrapper:
    def __init__(self, algo_instance):
        """
        Initialize wrapper with existing algo instance
        
        Args:
            algo_instance: Instance of the original betting algorithm
        """
        self.algo = algo_instance
        self.model = None
        self.scaler = StandardScaler()
        self.feature_cache = {}
        self.prediction_cache = {}
        
    def enhance_predictions(self):
        """
        Enhance the algo's market inefficiency detection with neural predictions
        """
        # Store original method
        original_detect = self.algo.detect_market_inefficiency
        
        def enhanced_detect(match_id):
            # Get original prediction
            original_prediction = original_detect(match_id)
            
            # If we have neural predictions, blend them
            if self.model is not None and match_id in self.prediction_cache:
                neural_probs = self.prediction_cache[match_id]
                
                # Blend predictions (70% neural, 30% original)
                blend_weight = 0.7
                
                blended_edge = (
                    neural_probs['edge'] * blend_weight +
                    original_prediction['edge'] * (1 - blend_weight)
                )
                
                blended_prob = (
                    neural_probs['true_prob'] * blend_weight +
                    original_prediction['true_prob'] * (1 - blend_weight)
                )
                
                return {
                    'edge': blended_edge,
                    'bet_type': neural_probs['bet_type'],
                    'bookie': original_prediction['bookie'],
                    'true_prob': blended_prob,
                    'implied_prob': original_prediction['implied_prob']
                }
            
            return original_prediction
        
        # Replace method with enhanced version
        self.algo.detect_market_inefficiency = enhanced_detect
    
    def prepare_features(self):
        """
        Prepare features for all matches in the algo's dataset
        """
        print("Preparing neural network features...")
        
        features_list = []
        match_ids = []
        
        for match_id in self.algo.historical_odds['match_id'].unique():
            match_data = self.algo.historical_odds[
                self.algo.historical_odds['match_id'] == match_id
            ].iloc[0]
            
            # Calculate team form features
            home_form = self._calculate_team_form(
                match_data['home_team'],
                match_data['match_date']
            )
            away_form = self._calculate_team_form(
                match_data['away_team'],
                match_data['match_date']
            )
            
            # Calculate market features
            market_features = self._calculate_market_features(match_id)
            
            # Combine all features
            feature_vector = np.concatenate([
                home_form,
                away_form,
                market_features
            ])
            
            features_list.append(feature_vector)
            match_ids.append(match_id)
            
            # Cache features for later use
            self.feature_cache[match_id] = feature_vector
        
        return np.array(features_list), match_ids
    
    def _calculate_team_form(self, team, date, lookback=5):
        """
        Calculate team form features
        """
        past_matches = self.algo.match_results[
            (self.algo.match_results['date'] < date) &
            ((self.algo.match_results['home_team'] == team) |
            (self.algo.match_results['away_team'] == team))
        ].tail(lookback)
        
        # Calculate win rate
        wins = len(past_matches[past_matches['winner'] == team])
        win_rate = wins / len(past_matches) if len(past_matches) > 0 else 0.5
        
        # Calculate recent results (weighted)
        weights = np.exp(np.linspace(-1, 0, len(past_matches)))
        weights /= weights.sum() if len(weights) > 0 else 1
        
        # Fixed version using pandas dataframe instead of itertuples
        weighted_performance = sum(
            weights[i] * (1 if row['winner'] == team else 0)
            for i, (_, row) in enumerate(past_matches.iterrows())
        ) if len(past_matches) > 0 else 0.5
        
        return np.array([win_rate, weighted_performance])

    
    def _calculate_market_features(self, match_id):
        """
        Calculate market-based features
        """
        match_odds = self.algo.historical_odds[
            self.algo.historical_odds['match_id'] == match_id
        ]
        
        # Calculate average odds
        avg_odds = match_odds[['home_win_odds', 'draw_odds', 'away_win_odds']].mean()
        
        # Calculate odds variation
        odds_std = match_odds[['home_win_odds', 'draw_odds', 'away_win_odds']].std()
        
        # Calculate implied probabilities
        total_prob = (1/avg_odds['home_win_odds'] + 
                     1/avg_odds['draw_odds'] + 
                     1/avg_odds['away_win_odds'])
        
        implied_probs = np.array([
            1/avg_odds['home_win_odds'] / total_prob,
            1/avg_odds['draw_odds'] / total_prob,
            1/avg_odds['away_win_odds'] / total_prob
        ])
        
        return np.concatenate([avg_odds, odds_std, implied_probs])
    
    def train_neural_network(self):
        """
        Train neural network on historical data
        """
        print("Training neural network...")
        
        # Prepare features and labels
        features, match_ids = self.prepare_features()
        
        # Prepare labels (actual results)
        labels = []
        for match_id in match_ids:
            result = self.algo.match_results[
                self.algo.match_results['match_id'] == match_id
            ].iloc[0]
            
            match_odds = self.algo.historical_odds[
                self.algo.historical_odds['match_id'] == match_id
            ].iloc[0]
            
            if result['winner'] == match_odds['home_team']:
                labels.append([1, 0, 0])
            elif result['winner'] == 'draw':
                labels.append([0, 1, 0])
            else:
                labels.append([0, 0, 1])
        
        labels = np.array(labels)
        
        # Scale features
        scaled_features = self.scaler.fit_transform(features)
        
        # Build model
        self.model = models.Sequential([
            layers.Dense(64, activation='relu', input_shape=(features.shape[1],)),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dense(3, activation='softmax')
        ])
        
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train model
        history = self.model.fit(
            scaled_features, labels,
            validation_split=0.2,
            epochs=50,
            batch_size=32,
            verbose=1
        )
        
        # Generate and cache predictions
        self._generate_predictions(scaled_features, match_ids)
        
        return history
    
    def _generate_predictions(self, scaled_features, match_ids):
        """
        Generate and cache predictions for all matches
        """
        predictions = self.model.predict(scaled_features)
        
        for match_id, probs in zip(match_ids, predictions):
            match_odds = self.algo.historical_odds[
                self.algo.historical_odds['match_id'] == match_id
            ].iloc[0]
            
            # Calculate implied probabilities
            total_prob = (1/match_odds['home_win_odds'] + 
                        1/match_odds['draw_odds'] + 
                        1/match_odds['away_win_odds'])
            
            implied_probs = {
                'home': (1/match_odds['home_win_odds']) / total_prob,
                'draw': (1/match_odds['draw_odds']) / total_prob,
                'away': (1/match_odds['away_win_odds']) / total_prob
            }
            
            # Find best edge
            edges = {
                'home': probs[0] - implied_probs['home'],
                'draw': probs[1] - implied_probs['draw'],
                'away': probs[2] - implied_probs['away']
            }
            
            best_bet = max(edges.items(), key=lambda x: x[1])
            
            self.prediction_cache[match_id] = {
                'edge': best_bet[1],
                'bet_type': best_bet[0],
                'true_prob': probs[['home', 'draw', 'away'].index(best_bet[0])]
            }
    
    def save_model(self, path='betting_nn_model'):
        """
        Save the trained model and scaler
        """
        if self.model is not None:
            self.model.save(f'{path}_keras')
            joblib.dump(self.scaler, f'{path}_scaler.pkl')
    
    def load_model(self, path='betting_nn_model'):
        """
        Load a previously trained model and scaler
        """
        try:
            self.model = models.load_model(f'{path}_keras')
            self.scaler = joblib.load(f'{path}_scaler.pkl')
            return True
        except:
            print("No saved model found. Please train the model first.")
            return False
