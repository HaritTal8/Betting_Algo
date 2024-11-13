# #understading the requirements for this file:
# project_directory/
# ├── betting_nn_wrapper.py  # Contains the BettingNeuralWrapper class
# ├── algo.py               # Contains your original betting algorithm
# ├── test.py              # Contains your BettingSystemTest class
# └── main.py              # The script that runs everything together

#here is the classes execution flow:
# Original Algo --> Neural Wrapper --> Enhanced Predictions --> Backtest


#for neural networks:
from algo import algo
from betting_nn_wrapper import BettingNeuralWrapper
from test import BettingSystemTest

# Create instance of original algorithm
betting_algo = algo()

# Create neural wrapper
nn_wrapper = BettingNeuralWrapper(betting_algo)

# Run your test as normal
test_suite = BettingSystemTest(initial_bankroll=10000)

# Before running the backtest, enhance the algorithm with neural predictions
nn_wrapper.train_neural_network()  # Train the neural network
nn_wrapper.enhance_predictions()    # Enhance the original algorithm

# Now run your test as normal - it will use the enhanced predictions
results = test_suite.run_backtest(
    start_date='2022-08-01',
    end_date='2024-01-01',
    min_edge=0.05
)

# Optional: Save the trained model for future use
nn_wrapper.save_model()
