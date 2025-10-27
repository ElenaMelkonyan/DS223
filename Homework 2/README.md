# Multi-Armed Bandit Algorithms for A/B Testing

## Overview

This project implements and compares two popular multi-armed bandit algorithms for A/B testing scenarios: **Epsilon Greedy** and **Thompson Sampling**. The implementation provides a comprehensive framework for evaluating bandit algorithms with detailed performance metrics, visualizations, and statistical analysis.

## Author
**Elena Melkonyan**  
*Marketing Analytics - Homework 2*

## Features

### Algorithms Implemented
- **Epsilon Greedy with Epsilon Decay**: Uses 1/t decay strategy for exploration-exploitation balance
- **Thompson Sampling**: Bayesian approach with Beta-Bernoulli approximation for known precision

### Performance Analysis
- Cumulative rewards and regrets tracking
- Arm selection frequency analysis
- Statistical comparison between algorithms
- Comprehensive visualization suite

### Visualizations
- **Performance Analysis Plots**: 6-panel visualization showing cumulative rewards, regrets, and arm selection patterns
- **Algorithm Comparison**: Side-by-side comparison of both algorithms
- **Multiple Scale Views**: Both linear and logarithmic scale plots
- **Real-time Metrics**: Average reward per trial analysis

## Project Structure

The project consists of the following files:

- **Bandit.py**: Main implementation file containing all algorithm classes and comparison functions
- **requirements.txt**: Python package dependencies
- **Homework 2.pdf**: Original assignment description and requirements
- **README.md**: This documentation file

## Dependencies

The project requires the following Python packages:

```
loguru==0.7.2
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.5.0
seaborn>=0.11.0
```

## Installation

To set up the project on your local machine:

1. Download or clone the project files to your desired directory
2. Install the required Python packages by running:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Quick Start

To run the complete comparison experiment, simply execute the main script:

```bash
python Bandit.py
```

This command will perform the following actions:
- Initialize both algorithms with 4 bandit arms having rewards [1, 2, 3, 4]
- Execute 20,000 trials for each algorithm
- Generate comprehensive visualizations and save them as PNG files
- Export detailed results to CSV files
- Display performance metrics in the terminal


## Algorithm Details

### Epsilon Greedy Algorithm

**Strategy**: Balances exploration and exploitation using epsilon decay
- **Exploration**: Random arm selection with probability ε = 1/t
- **Exploitation**: Select best arm (highest average reward) with probability (1-ε)
- **Decay**: Epsilon decreases over time (1/t), reducing exploration as learning progresses

**Key Features**:
- Simple and intuitive approach that is easy to understand and implement
- Guaranteed to explore all arms, ensuring no arm is completely ignored
- Converges to optimal strategy over time as epsilon decreases

### Thompson Sampling Algorithm

**Strategy**: Bayesian approach using posterior sampling
- **Prior**: Beta distribution for each arm (α=1, β=1)
- **Update**: Beta parameters updated based on binary reward outcomes
- **Selection**: Sample from posterior distributions and select arm with highest sample

**Key Features**:
- Naturally balances exploration and exploitation through probabilistic sampling
- Uses uncertainty in decision making, selecting arms based on their potential rather than just current estimates
- Often outperforms epsilon-greedy in practice due to more sophisticated exploration strategy

## Output Files

The program generates several output files:

### CSV Files
- **epsilon_greedy_results.csv**: Contains detailed trial-by-trial results for the Epsilon Greedy algorithm
- **thompson_sampling_results.csv**: Contains detailed trial-by-trial results for the Thompson Sampling algorithm  
- **combined_bandit_results.csv**: Contains results from both algorithms in a single file for easy comparison

### Visualization Files
- **bandit_performance_analysis.png**: Comprehensive 6-panel performance analysis showing multiple metrics
- **algorithm_comparison.png**: Side-by-side comparison of both algorithms' performance

## Performance Metrics

### Key Metrics Tracked
- **Total Reward**: Sum of all rewards received across all trials
- **Average Reward**: Mean reward per trial, indicating overall performance
- **Total Regret**: Cumulative difference from the optimal reward that could have been achieved
- **Average Regret**: Mean regret per trial, showing how much performance is lost compared to optimal
- **Arm Selection Counts**: Number of times each arm was selected, revealing exploration patterns
- **Final Parameter Values**: Learned parameters for each algorithm after all trials

### Regret Calculation
The regret is calculated as: **Regret = Optimal Reward - Actual Reward Received**

Where the optimal reward is the maximum possible reward from the best arm (in this case, 4 from Bandit 4). Lower regret values indicate better performance, as the algorithm is closer to achieving the optimal outcome.

## Visualization Features

### Performance Analysis Plot (`plot1`)
The comprehensive 6-panel visualization includes:
1. **Cumulative Rewards (Linear Scale)**: Shows how total rewards accumulate over time
2. **Cumulative Regrets (Linear Scale)**: Displays how total regrets build up over trials
3. **Arm Selection Frequency**: Bar chart showing how often each arm was selected
4. **Cumulative Rewards (Logarithmic Scale)**: Logarithmic view of rewards for better visualization of early performance
5. **Cumulative Regrets (Logarithmic Scale)**: Logarithmic view of regrets to highlight differences in early trials
6. **Average Reward per Trial**: Shows how the average reward converges over time

### Algorithm Comparison Plot (`plot2`)
The side-by-side comparison includes:
1. **Cumulative Rewards Comparison**: Direct comparison of how rewards accumulate for both algorithms
2. **Cumulative Regrets Comparison**: Direct comparison of how regrets accumulate for both algorithms

## Technical Implementation

### Class Structure
The implementation follows an object-oriented design with the following main classes:
- **`Bandit`**: Abstract base class that defines the common interface for all bandit algorithms
- **`EpsilonGreedy`**: Concrete implementation of the epsilon-greedy algorithm with decay
- **`ThompsonSampling`**: Concrete implementation of the Thompson sampling algorithm
- **`Visualization`**: Utility class providing comprehensive plotting and analysis tools

### Key Methods
Each algorithm implements the following core methods:
- **`pull(t)`**: Select which arm to pull for trial t
- **`update(arm, reward)`**: Update the algorithm's internal state based on the observed reward
- **`experiment()`**: Run the complete experiment for the specified number of trials
- **`report()`**: Generate performance metrics and save results to files

## Experimental Setup

### Default Configuration
The experiments are configured with the following default parameters:
- **Number of Arms**: 4 bandit arms
- **True Rewards**: [1, 2, 3, 4] for arms 1 through 4 respectively
- **Number of Trials**: 20,000 trials per algorithm
- **Noise**: Gaussian noise with standard deviation 0.1 is added to each reward
- **Random Seed**: 42 (ensures reproducible results across runs)

### Bandit Arms
The four bandit arms have the following true reward values:
- **Bandit 1**: True reward = 1 (lowest performing)
- **Bandit 2**: True reward = 2 (second lowest)
- **Bandit 3**: True reward = 3 (second highest)
- **Bandit 4**: True reward = 4 (optimal arm with highest reward)

## Expected Results

### Typical Performance
Based on the algorithm characteristics, you can expect the following performance patterns:
- **Thompson Sampling** typically achieves lower regret and higher total rewards due to its sophisticated exploration strategy
- **Epsilon Greedy** shows more systematic exploration in early trials due to its fixed epsilon schedule
- Both algorithms should gradually converge to selecting the optimal arm (Bandit 4) more frequently as they learn

### Learning Behavior
The learning process typically follows these phases:
- **Early trials (1-1000)**: High exploration phase with frequent suboptimal arm selection as algorithms gather information
- **Middle trials (1000-10000)**: Gradual shift toward the optimal arm as algorithms begin to identify the best choice
- **Late trials (10000+)**: Predominantly optimal arm selection with occasional exploration to maintain learning
---

*This README provides comprehensive documentation for the Multi-Armed Bandit A/B Testing implementation. The code is well-documented with extensive examples for both basic usage and advanced customization, making it suitable for educational and research purposes.*
