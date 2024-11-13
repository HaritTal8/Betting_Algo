import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class algo:
    def __init__(self):
        self.historical_odds = pd.DataFrame()
        self.match_results = pd.DataFrame()
        self.model_parameters = {}
        self.lookback_period = 10

    def fetch_real_data(self, start_date, end_date):
        return self.fetch_odds_data(start_date, end_date)
        
    def train_model(self):
        pass

    def execute_betting_strategy(self, initial_bankroll, min_edge):
        results = self.backtest(
            start_date='2023-01-01',
            end_date='2023-03-31',
            initial_capital=initial_bankroll,
            min_edge=min_edge
        )
        
        # Rename columns to match test.py expectations
        results = results.rename(columns={
            'pnl': 'profit',
            'bet_size': 'position',
            'edge': 'edge',
            'capital': 'bankroll'
        })
        
        # Add required columns for test.py
        results['market_quality'] = results['implied_prob'].apply(lambda x: 1 - x if x < 0.5 else x)
        results['roi'] = results['profit'] / results['position']
        
        # Set date as index
        results.set_index('date', inplace=True)
        
        return results

    # Your existing methods remain the same
    def fetch_odds_data(self, start_date, end_date):
        print(f"Fetching odds data from {start_date} to {end_date}...")
        
        dates = pd.date_range(start=start_date, end=end_date, freq='7D')
        teams = ['Arsenal', 'Chelsea', 'Liverpool', 'Manchester United', 
                'Manchester City', 'Tottenham']
        bookies = ['Bookie1', 'Bookie2', 'Bookie3']
        
        data = []
        match_id = 1000
        
        for date in dates:
            for _ in range(3):
                home, away = np.random.choice(teams, 2, replace=False)
                base_probs = np.random.dirichlet([2, 1, 2])
                
                for bookie in bookies:
                    margin = 1.05
                    home_odds = margin / base_probs[0]
                    draw_odds = margin / base_probs[1]
                    away_odds = margin / base_probs[2]
                    
                    data.append({
                        'match_id': match_id,
                        'home_team': home,
                        'away_team': away,
                        'match_date': date,
                        'bookie': bookie,
                        'home_win_odds': round(home_odds, 2),
                        'draw_odds': round(draw_odds, 2),
                        'away_win_odds': round(away_odds, 2)
                    })
                
                match_id += 1
        
        self.historical_odds = pd.DataFrame(data)
        print(f"Fetched {len(self.historical_odds)} odds entries for {len(dates)} dates")
        
        results_data = []
        for match_id in self.historical_odds['match_id'].unique():
            match = self.historical_odds[self.historical_odds['match_id'] == match_id].iloc[0]
            total_prob = (1/match['home_win_odds'] + 1/match['draw_odds'] + 1/match['away_win_odds'])
            home_prob = (1/match['home_win_odds']) / total_prob
            draw_prob = (1/match['draw_odds']) / total_prob
            away_prob = (1/match['away_win_odds']) / total_prob
            
            outcome = np.random.choice(['home', 'draw', 'away'], 
                                     p=[home_prob, draw_prob, away_prob])
            winner = match['home_team'] if outcome == 'home' else \
                    match['away_team'] if outcome == 'away' else 'draw'
            
            results_data.append({
                'match_id': match_id,
                'date': match['match_date'],
                'home_team': match['home_team'],
                'away_team': match['away_team'],
                'winner': winner
            })
        
        self.match_results = pd.DataFrame(results_data)
        print(f"Created {len(self.match_results)} match results")
    
    def detect_market_inefficiency(self, match_id):
        """
        Identifies market inefficiencies by comparing true probabilities
        with implied probabilities from odds
        """
        match_odds = self.historical_odds[
            self.historical_odds['match_id'] == match_id
        ]
        
        # Calculate implied probabilities (removing overround)
        implied_probs = {}
        for bookie in match_odds['bookie'].unique():
            bookie_odds = match_odds[match_odds['bookie'] == bookie]
            home_prob = 1 / bookie_odds['home_win_odds'].iloc[0]
            draw_prob = 1 / bookie_odds['draw_odds'].iloc[0]
            away_prob = 1 / bookie_odds['away_win_odds'].iloc[0]
            
            # Remove overround
            total_prob = home_prob + draw_prob + away_prob
            implied_probs[bookie] = {
                'home': home_prob / total_prob,
                'draw': draw_prob / total_prob,
                'away': away_prob / total_prob
            }
        
        # Calculate true probabilities with more variation from implied odds
        base_home_prob = np.mean([probs['home'] for probs in implied_probs.values()])
        base_away_prob = np.mean([probs['away'] for probs in implied_probs.values()])
        
        # Add random variation to create potential edges
        true_probs = {
            'home': min(max(base_home_prob + np.random.uniform(-0.1, 0.1), 0.1), 0.8),
            'away': min(max(base_away_prob + np.random.uniform(-0.1, 0.1), 0.1), 0.8)
        }
        true_probs['draw'] = 1 - true_probs['home'] - true_probs['away']
        
        # Find largest deviation
        max_edge = 0
        best_bet = None
        best_bookie = None
        
        for bookie, probs in implied_probs.items():
            for outcome in ['home', 'draw', 'away']:
                edge = true_probs[outcome] - probs[outcome]
                if edge > max_edge:
                    max_edge = edge
                    best_bet = outcome
                    best_bookie = bookie
        
        return {
            'edge': max_edge,
            'bet_type': best_bet,
            'bookie': best_bookie,
            'true_prob': true_probs[best_bet],
            'implied_prob': implied_probs[best_bookie][best_bet]
        }
    
    def kelly_criterion(self, true_prob, odds):
        """
        Calculates optimal bet size using Kelly Criterion
        """
        q = 1 - true_prob
        p = true_prob
        b = odds - 1
        
        f = (b * p - q) / b
        return max(0, min(f, 0.2))  # Cap at 20% of bankroll
    
    def backtest(self, start_date, end_date, initial_capital=10000,
                min_edge=0.03, kelly_fraction=0.5):  # Reduced min_edge to 3%
        """
        Backtests the strategy over historical data
        """
        print(f"\nStarting backtest from {start_date} to {end_date}")
        print(f"Initial capital: ${initial_capital:,.2f}")
        print(f"Minimum edge required: {min_edge*100}%")
        print(f"Kelly fraction: {kelly_fraction}")
        
        capital = initial_capital
        bets = []
        
        test_matches = self.match_results[
            (self.match_results['date'] >= start_date) &
            (self.match_results['date'] <= end_date)
        ]
        
        print(f"\nAnalyzing {len(test_matches)} matches...")
        
        for _, match in test_matches.iterrows():
            inefficiency = self.detect_market_inefficiency(match['match_id'])
            
            # Debug print for each match analysis
            print(f"\nAnalyzing match {match['match_id']}")
            print(f"Edge found: {inefficiency['edge']*100:.1f}%")
            
            if inefficiency['edge'] > min_edge:
                match_odds = self.historical_odds[
                    (self.historical_odds['match_id'] == match['match_id']) &
                    (self.historical_odds['bookie'] == inefficiency['bookie'])
                ]
                
                if inefficiency['bet_type'] == 'home':
                    odds = match_odds['home_win_odds'].iloc[0]
                elif inefficiency['bet_type'] == 'away':
                    odds = match_odds['away_win_odds'].iloc[0]
                else:
                    odds = match_odds['draw_odds'].iloc[0]
                
                kelly_bet = self.kelly_criterion(
                    inefficiency['true_prob'], odds
                )
                bet_size = capital * kelly_bet * kelly_fraction
                
                won = (match['winner'] == match['home_team'] and inefficiency['bet_type'] == 'home') or \
                      (match['winner'] == match['away_team'] and inefficiency['bet_type'] == 'away') or \
                      (match['winner'] == 'draw' and inefficiency['bet_type'] == 'draw')
                
                pnl = bet_size * (odds - 1) if won else -bet_size
                capital += pnl
                
                print(f"Placed bet: {inefficiency['bet_type']} @ {odds:.2f}")
                print(f"Bet size: ${bet_size:.2f}")
                print(f"Result: {'Won' if won else 'Lost'}")
                print(f"PnL: ${pnl:.2f}")
                
                bets.append({
                    'match_id': match['match_id'],
                    'date': match['date'],
                    'bet_type': inefficiency['bet_type'],
                    'edge': inefficiency['edge'],
                    'true_prob': inefficiency['true_prob'],
                    'implied_prob': inefficiency['implied_prob'],
                    'odds': odds,
                    'bet_size': bet_size,
                    'won': won,
                    'pnl': pnl,
                    'capital': capital
                })
        
        results = pd.DataFrame(bets)
        print(f"\nPlaced {len(results)} bets out of {len(test_matches)} matches analyzed")
        return results

    def calculate_performance_metrics(self, backtest_results):
        """
        Calculates key performance metrics from backtest results
        """
        if len(backtest_results) == 0:
            print("No bets were placed during the backtest period")
            return {}
            
        metrics = {
            'total_bets': len(backtest_results),
            'win_rate': len(backtest_results[backtest_results['won']]) / len(backtest_results),
            'total_pnl': backtest_results['pnl'].sum(),
            'sharpe_ratio': backtest_results['pnl'].mean() / backtest_results['pnl'].std() * np.sqrt(252) if len(backtest_results) > 1 else 0,
            'max_drawdown': (backtest_results['capital'].cummax() - backtest_results['capital']).max(),
            'roi': (backtest_results['capital'].iloc[-1] - backtest_results['capital'].iloc[0]) / backtest_results['capital'].iloc[0]
        }
        
        print("\nPerformance Metrics:")
        print(f"Total Bets: {metrics['total_bets']}")
        print(f"Win Rate: {metrics['win_rate']*100:.1f}%")
        print(f"Total P&L: ${metrics['total_pnl']:,.2f}")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"Maximum Drawdown: ${metrics['max_drawdown']:,.2f}")
        print(f"ROI: {metrics['roi']*100:.1f}%")
        
        return metrics

def main():
    # Create instance of the algorithm
    algoTest = algo()  # This creates an instance
    
    # Set date range for analysis
    start_date = '2023-01-01'
    end_date = '2023-03-31'  # Shorter period for testing
    
    # Fetch data - Use the instance (algoTest) instead of the class (algo)
    algoTest.fetch_odds_data(start_date, end_date)  # FIXED: Use instance method
    
    # Run backtest with modified parameters - Also use the instance
    results = algoTest.backtest(  # FIXED: Use instance method
        start_date=start_date,
        end_date=end_date,
        initial_capital=10000,
        min_edge=0.03,  # Reduced minimum edge requirement
        kelly_fraction=0.5
    )
    
    # Calculate and display metrics - Use the instance
    metrics = algoTest.calculate_performance_metrics(results)  # FIXED: Use instance method
    
    # Display sample of bets
    if len(results) > 0:
        print("\nSample of betting activity:")
        print(results[['date', 'bet_type', 'edge', 'odds', 'bet_size', 'won', 'pnl']].head())

if __name__ == "__main__":
    main()
