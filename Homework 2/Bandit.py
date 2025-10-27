"""
A/B Testing Implementation with Epsilon Greedy and Thompson Sampling Algorithms

This module implements multi-armed bandit algorithms for A/B testing scenarios.
It includes Epsilon Greedy with epsilon decay and Thompson Sampling with known precision.

Author: Elena Melkonyan
Date: 2025
"""
############################### IMPORTS
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger



class Bandit(ABC):
    ##==== DO NOT REMOVE ANYTHING FROM THIS CLASS ====##

    @abstractmethod
    def __init__(self, p):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def pull(self, t):
        pass

    @abstractmethod
    def update(self, arm, reward):
        pass

    @abstractmethod
    def experiment(self):
        pass

    @abstractmethod
    def report(self):
        # store data in csv
        # print average reward (use f strings to make it informative)
        # print average regret (use f strings to make it informative)
        pass

#--------------------------------------#


class Visualization():
    """
    Visualization class for bandit algorithm performance analysis.
    
    This class provides methods to visualize the learning process and compare
    different bandit algorithms through various plots.
    """
    
    def __init__(self):
        """Initialize the visualization class."""
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def plot1(self, eg_results, ts_results, bandit_rewards):
        """
        Visualize the performance of each bandit algorithm.
        
        Creates both linear and logarithmic scale plots showing:
        - Cumulative rewards over time
        - Cumulative regrets over time
        - Arm selection frequency
        
        Args:
            eg_results (dict): EpsilonGreedy experiment results
            ts_results (dict): ThompsonSampling experiment results
            bandit_rewards (list): True rewards for each bandit arm
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Bandit Algorithm Performance Analysis', fontsize=16, fontweight='bold')
        
        trials = range(len(eg_results['cumulative_rewards']))
        
        # Plot 1: Cumulative Rewards (Linear)
        axes[0, 0].plot(trials, eg_results['cumulative_rewards'], label='Epsilon Greedy', linewidth=2)
        axes[0, 0].plot(trials, ts_results['cumulative_rewards'], label='Thompson Sampling', linewidth=2)
        axes[0, 0].set_title('Cumulative Rewards (Linear Scale)')
        axes[0, 0].set_xlabel('Trials')
        axes[0, 0].set_ylabel('Cumulative Reward')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Cumulative Regrets (Linear)
        axes[0, 1].plot(trials, eg_results['cumulative_regrets'], label='Epsilon Greedy', linewidth=2)
        axes[0, 1].plot(trials, ts_results['cumulative_regrets'], label='Thompson Sampling', linewidth=2)
        axes[0, 1].set_title('Cumulative Regrets (Linear Scale)')
        axes[0, 1].set_xlabel('Trials')
        axes[0, 1].set_ylabel('Cumulative Regret')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Arm Selection Frequency
        eg_arms = np.array(eg_results['arms_selected'])
        ts_arms = np.array(ts_results['arms_selected'])
        
        eg_counts = [np.sum(eg_arms == i) for i in range(len(bandit_rewards))]
        ts_counts = [np.sum(ts_arms == i) for i in range(len(bandit_rewards))]
        
        x = np.arange(len(bandit_rewards))
        width = 0.35
        
        axes[0, 2].bar(x - width/2, eg_counts, width, label='Epsilon Greedy', alpha=0.8)
        axes[0, 2].bar(x + width/2, ts_counts, width, label='Thompson Sampling', alpha=0.8)
        axes[0, 2].set_title('Arm Selection Frequency')
        axes[0, 2].set_xlabel('Bandit Arms')
        axes[0, 2].set_ylabel('Selection Count')
        axes[0, 2].set_xticks(x)
        axes[0, 2].set_xticklabels([f'Bandit {i+1}\n(Reward: {bandit_rewards[i]})' for i in range(len(bandit_rewards))])
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Plot 4: Cumulative Rewards (Log Scale)
        axes[1, 0].semilogy(trials, eg_results['cumulative_rewards'], label='Epsilon Greedy', linewidth=2)
        axes[1, 0].semilogy(trials, ts_results['cumulative_rewards'], label='Thompson Sampling', linewidth=2)
        axes[1, 0].set_title('Cumulative Rewards (Log Scale)')
        axes[1, 0].set_xlabel('Trials')
        axes[1, 0].set_ylabel('Cumulative Reward (log scale)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 5: Cumulative Regrets (Log Scale)
        axes[1, 1].semilogy(trials, eg_results['cumulative_regrets'], label='Epsilon Greedy', linewidth=2)
        axes[1, 1].semilogy(trials, ts_results['cumulative_regrets'], label='Thompson Sampling', linewidth=2)
        axes[1, 1].set_title('Cumulative Regrets (Log Scale)')
        axes[1, 1].set_xlabel('Trials')
        axes[1, 1].set_ylabel('Cumulative Regret (log scale)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Plot 6: Average Reward per Trial
        eg_avg_rewards = eg_results['cumulative_rewards'] / (np.arange(len(eg_results['cumulative_rewards'])) + 1)
        ts_avg_rewards = ts_results['cumulative_rewards'] / (np.arange(len(ts_results['cumulative_rewards'])) + 1)
        
        axes[1, 2].plot(trials, eg_avg_rewards, label='Epsilon Greedy', linewidth=2)
        axes[1, 2].plot(trials, ts_avg_rewards, label='Thompson Sampling', linewidth=2)
        axes[1, 2].set_title('Average Reward per Trial')
        axes[1, 2].set_xlabel('Trials')
        axes[1, 2].set_ylabel('Average Reward')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('bandit_performance_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info("Performance visualization plots saved as 'bandit_performance_analysis.png'")
    
    def plot2(self, eg_results, ts_results):
        """
        Compare E-greedy and Thompson Sampling cumulative rewards and regrets.
        
        Creates side-by-side comparison plots for cumulative rewards and regrets.
        
        Args:
            eg_results (dict): EpsilonGreedy experiment results
            ts_results (dict): ThompsonSampling experiment results
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Algorithm Comparison: Epsilon Greedy vs Thompson Sampling', fontsize=16, fontweight='bold')
        
        trials = range(len(eg_results['cumulative_rewards']))
        
        # Plot 1: Cumulative Rewards Comparison
        axes[0].plot(trials, eg_results['cumulative_rewards'], label='Epsilon Greedy', linewidth=3, color='blue', alpha=0.8)
        axes[0].plot(trials, ts_results['cumulative_rewards'], label='Thompson Sampling', linewidth=3, color='red', alpha=0.8)
        axes[0].set_title('Cumulative Rewards Comparison', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Number of Trials', fontsize=12)
        axes[0].set_ylabel('Cumulative Reward', fontsize=12)
        axes[0].legend(fontsize=12)
        axes[0].grid(True, alpha=0.3)
        
        # Add final values as text
        eg_final = eg_results['cumulative_rewards'][-1]
        ts_final = ts_results['cumulative_rewards'][-1]
        axes[0].text(0.02, 0.98, f'Final Reward:\nEpsilon Greedy: {eg_final:.2f}\nThompson Sampling: {ts_final:.2f}', 
                    transform=axes[0].transAxes, verticalalignment='top', 
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Plot 2: Cumulative Regrets Comparison
        axes[1].plot(trials, eg_results['cumulative_regrets'], label='Epsilon Greedy', linewidth=3, color='blue', alpha=0.8)
        axes[1].plot(trials, ts_results['cumulative_regrets'], label='Thompson Sampling', linewidth=3, color='red', alpha=0.8)
        axes[1].set_title('Cumulative Regrets Comparison', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Number of Trials', fontsize=12)
        axes[1].set_ylabel('Cumulative Regret', fontsize=12)
        axes[1].legend(fontsize=12)
        axes[1].grid(True, alpha=0.3)
        
        # Add final values as text
        eg_regret = eg_results['cumulative_regrets'][-1]
        ts_regret = ts_results['cumulative_regrets'][-1]
        axes[1].text(0.02, 0.98, f'Final Regret:\nEpsilon Greedy: {eg_regret:.2f}\nThompson Sampling: {ts_regret:.2f}', 
                    transform=axes[1].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('algorithm_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info("Algorithm comparison plots saved as 'algorithm_comparison.png'")

#--------------------------------------#

class EpsilonGreedy(Bandit):
    """
    Epsilon Greedy algorithm implementation for multi-armed bandit problems.
    
    This class implements the epsilon-greedy strategy with epsilon decay (1/t).
    It balances exploration and exploitation by choosing the best arm with 
    probability (1-epsilon) and exploring randomly with probability epsilon.
    
    Attributes:
        bandit_rewards (list): List of true rewards for each bandit arm
        n_arms (int): Number of bandit arms
        epsilon (float): Initial exploration rate
        n_trials (int): Number of trials to run
        counts (np.array): Number of times each arm was pulled
        values (np.array): Average reward for each arm
        rewards_history (list): History of rewards received
        arms_selected (list): History of arms selected
        cumulative_rewards (np.array): Cumulative rewards over time
        cumulative_regrets (np.array): Cumulative regrets over time
    """
    
    def __init__(self, bandit_rewards, n_trials=20000, epsilon=0.1):
        """
        Initialize the Epsilon Greedy bandit algorithm.
        
        Args:
            bandit_rewards (list): True rewards for each bandit arm
            n_trials (int): Number of trials to run
            epsilon (float): Initial exploration rate
        """
        self.bandit_rewards = np.array(bandit_rewards)
        self.n_arms = len(bandit_rewards)
        self.epsilon = epsilon
        self.n_trials = n_trials
        
        # Initialize tracking variables
        self.counts = np.zeros(self.n_arms)
        self.values = np.zeros(self.n_arms)
        self.rewards_history = []
        self.arms_selected = []
        self.cumulative_rewards = np.zeros(n_trials)
        self.cumulative_regrets = np.zeros(n_trials)
        
        # Calculate optimal reward for regret calculation
        self.optimal_reward = np.max(self.bandit_rewards)
        
        logger.info(f"Initialized EpsilonGreedy with {self.n_arms} arms, epsilon={epsilon}, trials={n_trials}")
    
    def __repr__(self):
        """String representation of the EpsilonGreedy instance."""
        return f"EpsilonGreedy(n_arms={self.n_arms}, epsilon={self.epsilon}, trials={self.n_trials})"
    
    def pull(self, t):
        # epsilon decay: 1/t
        current_epsilon = 1.0 / t
        
        if np.random.random() < current_epsilon:
            # explore
            arm = np.random.randint(0, self.n_arms)
        else:
            # exploit
            arm = np.argmax(self.values)
        
        return arm
    
    def update(self, arm, reward):
        self.counts[arm] += 1
        n = self.counts[arm]
        value = self.values[arm]
        
        # update average
        self.values[arm] = value + (reward - value) / n
        
        # keep track
        self.rewards_history.append(reward)
        self.arms_selected.append(arm)
    
    def experiment(self):
        logger.info("Starting EpsilonGreedy experiment...")
        
        for t in range(self.n_trials):
            arm = self.pull(t + 1)  # t+1 because we need 1-indexed
            
            # get reward with some noise
            true_reward = self.bandit_rewards[arm]
            noise = np.random.normal(0, 0.1)
            reward = max(0, true_reward + noise)
            
            self.update(arm, reward)
            
            # update cumulative stuff
            if t == 0:
                self.cumulative_rewards[t] = reward
                self.cumulative_regrets[t] = self.optimal_reward - true_reward
            else:
                self.cumulative_rewards[t] = self.cumulative_rewards[t-1] + reward
                self.cumulative_regrets[t] = self.cumulative_regrets[t-1] + (self.optimal_reward - true_reward)
        
        logger.info("EpsilonGreedy experiment completed")
        
        return {
            'rewards': self.rewards_history,
            'arms_selected': self.arms_selected,
            'cumulative_rewards': self.cumulative_rewards,
            'cumulative_regrets': self.cumulative_regrets,
            'final_values': self.values.copy(),
            'final_counts': self.counts.copy()
        }
    
    def report(self):
        total_reward = np.sum(self.rewards_history)
        avg_reward = np.mean(self.rewards_history)
        total_regret = self.cumulative_regrets[-1]
        avg_regret = total_regret / self.n_trials
        
        # save to csv
        data = {
            'Bandit': [f'Bandit_{i+1}' for i in self.arms_selected],
            'Reward': self.rewards_history,
            'Algorithm': ['EpsilonGreedy'] * len(self.rewards_history)
        }
        df = pd.DataFrame(data)
        df.to_csv('epsilon_greedy_results.csv', index=False)
        
        # print results
        logger.info(f"EpsilonGreedy Results:")
        logger.info(f"Total Reward: {total_reward:.2f}")
        logger.info(f"Average Reward: {avg_reward:.4f}")
        logger.info(f"Total Regret: {total_regret:.2f}")
        logger.info(f"Average Regret: {avg_regret:.4f}")
        logger.info(f"Final Arm Values: {self.values}")
        logger.info(f"Arm Selection Counts: {self.counts}")
        
        return {
            'total_reward': total_reward,
            'avg_reward': avg_reward,
            'total_regret': total_regret,
            'avg_regret': avg_regret,
            'final_values': self.values,
            'arm_counts': self.counts
        }

#--------------------------------------#

class ThompsonSampling(Bandit):
    """
    Thompson Sampling algorithm implementation for multi-armed bandit problems.
    
    This class implements Thompson Sampling with known precision (inverse variance).
    It uses Bayesian inference to balance exploration and exploitation by sampling
    from the posterior distribution of each arm's reward.
    
    Attributes:
        bandit_rewards (list): List of true rewards for each bandit arm
        n_arms (int): Number of bandit arms
        n_trials (int): Number of trials to run
        precision (float): Known precision (inverse variance) of rewards
        alpha (np.array): Alpha parameters for Beta distribution
        beta (np.array): Beta parameters for Beta distribution
        rewards_history (list): History of rewards received
        arms_selected (list): History of arms selected
        cumulative_rewards (np.array): Cumulative rewards over time
        cumulative_regrets (np.array): Cumulative regrets over time
    """
    
    def __init__(self, bandit_rewards, n_trials=20000, precision=1.0):
        """
        Initialize the Thompson Sampling bandit algorithm.
        
        Args:
            bandit_rewards (list): True rewards for each bandit arm
            n_trials (int): Number of trials to run
            precision (float): Known precision (inverse variance) of rewards
        """
        self.bandit_rewards = np.array(bandit_rewards)
        self.n_arms = len(bandit_rewards)
        self.n_trials = n_trials
        self.precision = precision
        
        # Initialize Beta distribution parameters (conjugate prior for Bernoulli)
        # For Thompson Sampling with Gaussian rewards, we use Beta-Bernoulli approximation
        self.alpha = np.ones(self.n_arms)  # Prior successes
        self.beta = np.ones(self.n_arms)   # Prior failures
        
        # Initialize tracking variables
        self.rewards_history = []
        self.arms_selected = []
        self.cumulative_rewards = np.zeros(n_trials)
        self.cumulative_regrets = np.zeros(n_trials)
        
        # Calculate optimal reward for regret calculation
        self.optimal_reward = np.max(self.bandit_rewards)
        
        logger.info(f"Initialized ThompsonSampling with {self.n_arms} arms, precision={precision}, trials={n_trials}")
    
    def __repr__(self):
        """String representation of the ThompsonSampling instance."""
        return f"ThompsonSampling(n_arms={self.n_arms}, precision={self.precision}, trials={self.n_trials})"
    
    def pull(self, t):
        # sample from beta distribution
        sampled_values = np.random.beta(self.alpha, self.beta)
        
        # pick the best one
        arm = np.argmax(sampled_values)
        
        return arm
    
    def update(self, arm, reward):
        # convert to binary for beta-bernoulli
        normalized_reward = min(1.0, max(0.0, reward / self.optimal_reward))
        
        # update beta params
        if normalized_reward > 0.5:
            self.alpha[arm] += 1
        else:
            self.beta[arm] += 1
        
        # keep track
        self.rewards_history.append(reward)
        self.arms_selected.append(arm)
    
    def experiment(self):
        logger.info("Starting ThompsonSampling experiment...")
        
        for t in range(self.n_trials):
            arm = self.pull(t + 1)  # t+1 because we need 1-indexed
            
            # get reward with some noise
            true_reward = self.bandit_rewards[arm]
            noise = np.random.normal(0, 0.1)
            reward = max(0, true_reward + noise)
            
            self.update(arm, reward)
            
            # update cumulative stuff
            if t == 0:
                self.cumulative_rewards[t] = reward
                self.cumulative_regrets[t] = self.optimal_reward - true_reward
            else:
                self.cumulative_rewards[t] = self.cumulative_rewards[t-1] + reward
                self.cumulative_regrets[t] = self.cumulative_regrets[t-1] + (self.optimal_reward - true_reward)
        
        logger.info("ThompsonSampling experiment completed")
        
        return {
            'rewards': self.rewards_history,
            'arms_selected': self.arms_selected,
            'cumulative_rewards': self.cumulative_rewards,
            'cumulative_regrets': self.cumulative_regrets,
            'final_alpha': self.alpha.copy(),
            'final_beta': self.beta.copy()
        }
    
    def report(self):
        total_reward = np.sum(self.rewards_history)
        avg_reward = np.mean(self.rewards_history)
        total_regret = self.cumulative_regrets[-1]
        avg_regret = total_regret / self.n_trials
        
        # save to csv
        data = {
            'Bandit': [f'Bandit_{i+1}' for i in self.arms_selected],
            'Reward': self.rewards_history,
            'Algorithm': ['ThompsonSampling'] * len(self.rewards_history)
        }
        df = pd.DataFrame(data)
        df.to_csv('thompson_sampling_results.csv', index=False)
        
        # print results
        logger.info(f"ThompsonSampling Results:")
        logger.info(f"Total Reward: {total_reward:.2f}")
        logger.info(f"Average Reward: {avg_reward:.4f}")
        logger.info(f"Total Regret: {total_regret:.2f}")
        logger.info(f"Average Regret: {avg_regret:.4f}")
        logger.info(f"Final Alpha Parameters: {self.alpha}")
        logger.info(f"Final Beta Parameters: {self.beta}")
        
        return {
            'total_reward': total_reward,
            'avg_reward': avg_reward,
            'total_regret': total_regret,
            'avg_regret': avg_regret,
            'final_alpha': self.alpha,
            'final_beta': self.beta
        }




def comparison():
    logger.info("Starting comparison...")
    
    # set seed for reproducibility
    np.random.seed(42)
    
    # setup
    bandit_rewards = [1, 2, 3, 4]
    n_trials = 20000
    
    # create algorithms
    eg_algorithm = EpsilonGreedy(bandit_rewards, n_trials, epsilon=0.1)
    ts_algorithm = ThompsonSampling(bandit_rewards, n_trials, precision=1.0)
    
    # run experiments
    logger.info("Running Epsilon Greedy...")
    eg_results = eg_algorithm.experiment()
    eg_metrics = eg_algorithm.report()
    
    logger.info("Running Thompson Sampling...")
    ts_results = ts_algorithm.experiment()
    ts_metrics = ts_algorithm.report()
    
    # make plots
    viz = Visualization()
    logger.info("Making plots...")
    viz.plot1(eg_results, ts_results, bandit_rewards)
    viz.plot2(eg_results, ts_results)
    
    # combine data for csv
    combined_data = []
    for i, (bandit, reward) in enumerate(zip(eg_results['arms_selected'], eg_results['rewards'])):
        combined_data.append({
            'Bandit': f'Bandit_{bandit+1}',
            'Reward': reward,
            'Algorithm': 'EpsilonGreedy'
        })
    
    for i, (bandit, reward) in enumerate(zip(ts_results['arms_selected'], ts_results['rewards'])):
        combined_data.append({
            'Bandit': f'Bandit_{bandit+1}',
            'Reward': reward,
            'Algorithm': 'ThompsonSampling'
        })
    
    combined_df = pd.DataFrame(combined_data)
    combined_df.to_csv('combined_bandit_results.csv', index=False)
    
    # Print final comparison summary
    logger.info("="*60)
    logger.info("FINAL COMPARISON SUMMARY")
    logger.info("="*60)
    logger.info(f"Epsilon Greedy:")
    logger.info(f"  Total Reward: {eg_metrics['total_reward']:.2f}")
    logger.info(f"  Average Reward: {eg_metrics['avg_reward']:.4f}")
    logger.info(f"  Total Regret: {eg_metrics['total_regret']:.2f}")
    logger.info(f"  Average Regret: {eg_metrics['avg_regret']:.4f}")
    logger.info("")
    logger.info(f"Thompson Sampling:")
    logger.info(f"  Total Reward: {ts_metrics['total_reward']:.2f}")
    logger.info(f"  Average Reward: {ts_metrics['avg_reward']:.4f}")
    logger.info(f"  Total Regret: {ts_metrics['total_regret']:.2f}")
    logger.info(f"  Average Regret: {ts_metrics['avg_regret']:.4f}")
    logger.info("")
    
    # who won?
    if eg_metrics['total_reward'] > ts_metrics['total_reward']:
        logger.info("Epsilon Greedy won!")
    elif ts_metrics['total_reward'] > eg_metrics['total_reward']:
        logger.info("Thompson Sampling won!")
    else:
        logger.info("Tie!")
    
    if eg_metrics['total_regret'] < ts_metrics['total_regret']:
        logger.info("Epsilon Greedy has lower regret!")
    elif ts_metrics['total_regret'] < eg_metrics['total_regret']:
        logger.info("Thompson Sampling has lower regret!")
    else:
        logger.info("Same regret!")
    
    logger.info("="*60)
    logger.info("Done! Check the files.")
    
    return {
        'epsilon_greedy': eg_metrics,
        'thompson_sampling': ts_metrics,
        'eg_results': eg_results,
        'ts_results': ts_results
    }


if __name__ == '__main__':
    # setup logger
    logger.remove()
    logger.add(lambda msg: print(msg, end=""), level="INFO")
    
    logger.info("Starting A/B Testing Bandit Algorithm Comparison")
    logger.info("="*60)
    
    # run everything
    results = comparison()
    
    logger.info("Done! Check these files:")
    logger.info("  - epsilon_greedy_results.csv")
    logger.info("  - thompson_sampling_results.csv") 
    logger.info("  - combined_bandit_results.csv")
    logger.info("  - bandit_performance_analysis.png")
    logger.info("  - algorithm_comparison.png")


"""
BONUS: BETTER IMPLEMENTATION SUGGESTIONS

1. STATISTICAL ANALYSIS ENHANCEMENTS:
   - Add statistical significance testing between algorithms using t-tests or Mann-Whitney U tests
   - Implement confidence intervals for performance metrics to quantify uncertainty
   - Add bootstrap resampling for robust performance estimation

2. VISUALIZATION IMPROVEMENTS:
   - Add interactive plots using Plotly or Bokeh for better user experience
   - Implement real-time visualization during experiments to monitor progress
   - Add heatmaps for arm selection patterns over time to identify learning behavior
"""
