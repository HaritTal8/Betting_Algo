import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import seaborn as sns
from typing import Tuple, Dict
from tqdm import tqdm
from algo import algo  # Add this import at the top

class BettingSystemTest:
    def __init__(self, initial_bankroll: float = 10000):
        self.algo = algo()  # Now this will work
        self.initial_bankroll = initial_bankroll
        self.results = None
        
    def run_backtest(self, 
                    start_date: str = '2022-08-01',
                    end_date: str = '2024-01-01',
                    min_edge: float = 0.05) -> pd.DataFrame:
        """
        Runs a comprehensive backtest of the betting system
        """
        print("Starting backtest...")
        print("1. Fetching market data...")
        
        # Fetch and prepare data
        self.algo.fetch_real_data(start_date, end_date)
        
        print("2. Training prediction model...")
        self.algo.train_model()
        
        print("3. Executing betting strategy...")
        self.results = self.algo.execute_betting_strategy(
            initial_bankroll=self.initial_bankroll,
            min_edge=min_edge
        )
        
        print("4. Analyzing results...")
        self._analyze_results()
        
        return self.results
    
    def _analyze_results(self):
        """
        Performs detailed analysis of betting results
        """
        if self.results is None or len(self.results) == 0:
            print("No results to analyze. Run backtest first.")
            return
            
        # Calculate key metrics
        total_bets = len(self.results)
        winning_bets = len(self.results[self.results['profit'] > 0])
        win_rate = winning_bets / total_bets
        
        total_profit = self.results['profit'].sum()
        total_turnover = self.results['position'].sum()
        roi = (total_profit / total_turnover) * 100
        
        max_drawdown = self._calculate_max_drawdown()
        sharpe_ratio = self._calculate_sharpe_ratio()
        
        # Print summary
        print("\nBacktest Results Summary:")
        print(f"Period: {self.results.index.min()} to {self.results.index.max()}")
        print(f"Total Bets: {total_bets}")
        print(f"Win Rate: {win_rate:.2%}")
        print(f"ROI: {roi:.2%}")
        print(f"Total Profit: £{total_profit:,.2f}")
        print(f"Final Bankroll: £{self.initial_bankroll + total_profit:,.2f}")
        print(f"Max Drawdown: {max_drawdown:.2%}")
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        
        # Create visualizations
        self._plot_results()
    
    def _calculate_max_drawdown(self) -> float:
        """
        Calculates maximum drawdown from peak
        """
        bankroll = self.results['bankroll']
        rolling_max = bankroll.expanding(min_periods=1).max()
        drawdowns = bankroll / rolling_max - 1
        return abs(drawdowns.min())
    
    def _calculate_sharpe_ratio(self) -> float:
        """
        Calculates Sharpe ratio of betting returns
        """
        returns = self.results['profit'] / self.results['position']
        if len(returns) < 2:
            return 0
        return returns.mean() / returns.std() * np.sqrt(365)  # Annualized
    
    def _plot_results(self):
        """
        Creates visualization of betting performance
        """
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Bankroll Evolution
        plt.subplot(2, 2, 1)
        plt.plot(self.results.index, self.results['bankroll'])
        plt.title('Bankroll Evolution')
        plt.xlabel('Date')
        plt.ylabel('Bankroll (£)')
        
        # Plot 2: ROI Distribution
        plt.subplot(2, 2, 2)
        sns.histplot(self.results['roi'])
        plt.title('ROI Distribution')
        plt.xlabel('ROI')
        
        # Plot 3: Edge vs. Actual Returns
        plt.subplot(2, 2, 3)
        plt.scatter(self.results['edge'], self.results['roi'])
        plt.title('Edge vs. Actual Returns')
        plt.xlabel('Predicted Edge')
        plt.ylabel('Actual ROI')
        
        # Plot 4: Market Quality vs. ROI
        plt.subplot(2, 2, 4)
        plt.scatter(self.results['market_quality'], self.results['roi'])
        plt.title('Market Quality vs. ROI')
        plt.xlabel('Market Quality')
        plt.ylabel('ROI')
        
        plt.tight_layout()
        plt.show()
    
    def run_monte_carlo(self, n_simulations: int = 1000) -> Dict:
        """
        Runs Monte Carlo simulation to estimate risk metrics
        """
        if self.results is None:
            print("No results to simulate. Run backtest first.")
            return {}
            
        print("Running Monte Carlo simulation...")
        
        # Extract bet characteristics
        win_rate = len(self.results[self.results['profit'] > 0]) / len(self.results)
        avg_win = self.results[self.results['profit'] > 0]['profit'].mean()
        avg_loss = abs(self.results[self.results['profit'] < 0]['profit'].mean())
        
        # Run simulations
        final_bankrolls = []
        max_drawdowns = []
        
        for _ in tqdm(range(n_simulations)):
            bankroll = self.initial_bankroll
            peak = bankroll
            sim_bankroll = [bankroll]
            
            for _ in range(len(self.results)):
                if np.random.random() < win_rate:
                    bankroll += avg_win
                else:
                    bankroll -= avg_loss
                    
                peak = max(peak, bankroll)
                sim_bankroll.append(bankroll)
                
            final_bankrolls.append(bankroll)
            max_drawdowns.append((peak - min(sim_bankroll)) / peak)
        
        # Calculate risk metrics
        risk_metrics = {
            'expected_final_bankroll': np.mean(final_bankrolls),
            'bankroll_5th_percentile': np.percentile(final_bankrolls, 5),
            'bankroll_95th_percentile': np.percentile(final_bankrolls, 95),
            'expected_max_drawdown': np.mean(max_drawdowns),
            'worst_drawdown_95th': np.percentile(max_drawdowns, 95)
        }
        
        # Plot distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(final_bankrolls)
        plt.title('Distribution of Simulated Final Bankrolls')
        plt.xlabel('Final Bankroll (£)')
        plt.axvline(self.initial_bankroll, color='r', linestyle='--', 
                   label='Initial Bankroll')
        plt.legend()
        plt.show()
        
        return risk_metrics

def test_betting_system():
    """
    Main function to test the betting system
    """
    # Initialize test suite
    test_suite = BettingSystemTest(initial_bankroll=10000)
    
    # Run backtest
    results = test_suite.run_backtest(
        start_date='2022-08-01',
        end_date='2024-01-01',
        min_edge=0.05
    )
    
    # Run Monte Carlo simulation
    risk_metrics = test_suite.run_monte_carlo(n_simulations=1000)
    
    # Print risk metrics
    print("\nMonte Carlo Risk Analysis:")
    print(f"Expected Final Bankroll: £{risk_metrics['expected_final_bankroll']:,.2f}")
    print(f"5th Percentile Bankroll: £{risk_metrics['bankroll_5th_percentile']:,.2f}")
    print(f"95th Percentile Bankroll: £{risk_metrics['bankroll_95th_percentile']:,.2f}")
    print(f"Expected Max Drawdown: {risk_metrics['expected_max_drawdown']:.2%}")
    print(f"95th Percentile Worst Drawdown: {risk_metrics['worst_drawdown_95th']:.2%}")
    
    return results, risk_metrics

if __name__ == "__main__":
    results, risk_metrics = test_betting_system()
