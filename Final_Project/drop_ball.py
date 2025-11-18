"""
Filipino Drop Ball Gambling Game Simulation
============================================
A probabilistic simulation of the traditional Filipino Drop Ball game where balls
drop onto six cards (9, 10, J, Q, K, A).

This module implements two game models:
1. Fair Game: Equal probability (1/6) per card, multi-ball payout structure
2. Tweaked Game: Weighted probabilities + reduced payout (House Edge)

Game Rules:
- 3 balls drop one by one (not simultaneously)
- Fair Game Payout (adjusted for EV = 0):
  * 1 hit: 2.667x bet (net profit: +$1.667)
  * 2 hits: 4.0x bet (net profit: +$3.00)
  * 3 hits: 6.0x bet (net profit: +$5.00)
- Tweaked Game: Reduced payouts + weighted probabilities

Simulation Size: Configurable trials per model for robust statistical analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List

cards = ['9', '10', 'J', 'Q', 'K', 'A']
cards_with_joker = ['9', '10', 'J', 'Q', 'K', 'A', 'Joker']  # Tweaked game includes Joker
NumDrop = 1000  # Number of rounds (each round = 3 balls)
bet = 1.0
chosenCard = '10'
BALLS_PER_ROUND = 3  # Number of balls per round

class FairDropBall:
    """
    Fair Drop Ball game model with equal probability for all cards.
    
    Attributes:
        cards: List of card values in the game
        probabilities: Equal probability distribution for all cards
        balls_per_round: Number of balls that drop per round
        payout_structure: Dictionary mapping hits to payout multipliers
    """
    
    def __init__(self, cards: list, balls_per_round: int = 3):
        """
        Initialize the Fair Drop Ball game.
        
        Args:
            cards: List of card values
            balls_per_round: Number of balls per round (default: 3)
        """
        self.cards = cards
        self.balls_per_round = balls_per_round
        self.probabilities = np.ones(len(cards)) / len(cards)
        # Payout structure: hits -> multiplier (Adjusted for EV = 0)
        # Calculated to make Expected Value = 0 with p=1/6 and n=3 balls
        self.payout_structure = {
            0: 0,      # No hits: lose bet
            1: 2.0,  # 1 hit: 2.5x bet (net profit: +$1.50)
            2: 4.0,    # 2 hits: 4.0x bet (net profit: +$3.00)
            3: 6.0     # 3 hits: 6.0x bet (net profit: +$5.00)
        }
    
    def drop_ball(self) -> str:
        """
        Simulate a single ball drop with equal probability.
        
        Returns:
            The card where the ball landed
        """
        return np.random.choice(self.cards, p=self.probabilities)
    
    def drop_multiple_balls(self, chosen_card: str) -> tuple[List[str], int]:
        """
        Drop multiple balls and count hits on chosen card.
        
        Args:
            chosen_card: The card player bet on
            
        Returns:
            Tuple of (list of landed cards, number of hits)
        """
        landed_cards = [self.drop_ball() for _ in range(self.balls_per_round)]
        hits = sum(1 for card in landed_cards if card == chosen_card)
        return landed_cards, hits
    
    def calculate_payout(self, bet: float, hits: int) -> float:
        """
        Calculate the payout based on number of hits.
        
        Args:
            bet: Amount wagered
            hits: Number of balls that hit the chosen card
            
        Returns:
            Net profit/loss (positive for win, negative for loss)
        """
        if hits == 0:
            return -bet
        else:
            multiplier = self.payout_structure[hits]
            return bet * (multiplier - 1)  # Net profit
        
class TweakedDropBall(FairDropBall):
    """
    Tweaked Drop Ball game model with Joker card that causes automatic loss.
    Inherits from FairDropBall but adds a Joker card - landing on Joker = lose bet.
    
    Attributes:
        cards: List of card values including Joker (overridden)
        probabilities: Equal probability distribution for all cards including Joker (overridden)
        balls_per_round: Number of balls that drop per round (inherited)
        payout_structure: Same payout structure as Fair game (inherited)
    """
    
    def __init__(self, cards: list, balls_per_round: int = 3):
        """
        Initialize the Tweaked Drop Ball game with Joker card.
        
        Args:
            cards: List of card values (should include Joker)
            balls_per_round: Number of balls per round (default: 3)
        """
        # Initialize parent class
        super().__init__(cards, balls_per_round)
        
        # Override probabilities: Equal probability for all cards including Joker (1/7 each)
        self.probabilities = np.ones(len(cards)) / len(cards)
    
    def drop_multiple_balls(self, chosen_card: str) -> tuple[List[str], int]:
        """
        Drop multiple balls and count hits on chosen card.
        Joker doesn't count as a hit OR a loss, just reduces available balls.
        
        Args:
            chosen_card: The card player bet on
            
        Returns:
            Tuple of (list of landed cards, number of hits)
        """
        landed_cards = [self.drop_ball() for _ in range(self.balls_per_round)]
        
        # Count hits, ignoring Joker cards (Joker is a "dead ball")
        hits = sum(1 for card in landed_cards if card == chosen_card)
        return landed_cards, hits


def calculate_theoretical_ev(game_model, bet: float = 1.0) -> Dict:
    """
    Calculate the theoretical Expected Value for the game.
    
    Args:
        game_model: Instance of FairDropBall or TweakedDropBall
        bet: Bet amount (default: 1.0)
        
    Returns:
        Dictionary with EV breakdown by outcome
    """
    p = game_model.probabilities[game_model.cards.index('A')]  # Probability of hitting chosen card
    n = game_model.balls_per_round
    
    # Calculate binomial probabilities for each outcome
    outcomes = {}
    total_ev = 0
    
    for hits in range(n + 1):
        # Binomial probability: P(X=k) = C(n,k) * p^k * (1-p)^(n-k)
        from math import comb
        prob = comb(n, hits) * (p ** hits) * ((1 - p) ** (n - hits))
        payout = game_model.calculate_payout(bet, hits)
        ev_contribution = prob * payout
        
        outcomes[hits] = {
            'probability': prob,
            'payout': payout,
            'ev_contribution': ev_contribution
        }
        total_ev += ev_contribution
    
    return {
        'outcomes': outcomes,
        'total_ev': total_ev,
        'p_win': p,
        'balls_per_round': n
    }



def PlayDropBallGame(game_model, chosen_card: str, num_rounds: int, seed: int = None, verbose: bool = False, detailed: bool = False) -> Dict[str, float]:
    """
    Simulate multiple rounds of the ball drop game.
    
    Args:
        game_model: Instance of FairDropBall or TweakedDropBall
        chosen_card: Card the player bets on
        num_rounds: Number of rounds to simulate (each round = 3 balls)
        seed: Random seed for reproducibility (default: None)
        verbose: If True, print details of each round (default: False)
        detailed: If True, return detailed round-by-round data for visualization (default: False)
    """
    if seed is not None:
        np.random.seed(seed)
    
    results = []
    hits_distribution = {0: 0, 1: 0, 2: 0, 3: 0}
    
    # Track detailed data if requested
    if detailed:
        round_data = []
        cumulative_profit = []
        card_frequencies = {card: 0 for card in game_model.cards}
        # Track cards that got exactly 1, 2, or 3 balls in a single round
        single_hit_cards = {card: 0 for card in game_model.cards}
        double_hit_cards = {card: 0 for card in game_model.cards}
        triple_hit_cards = {card: 0 for card in game_model.cards}
    
    for round_num in range(1, num_rounds + 1):
        landed_cards, hits = game_model.drop_multiple_balls(chosen_card)
        payout = game_model.calculate_payout(bet, hits)
        results.append(payout)
        hits_distribution[hits] += 1
        
        if detailed:
            # Track cumulative profit
            cumulative_profit.append(sum(results))
            
            # Count frequency of each card across all balls
            for card in landed_cards:
                card_frequencies[card] += 1
            
            # Count how many times each card appeared in this round
            round_card_counts = {}
            for card in landed_cards:
                round_card_counts[card] = round_card_counts.get(card, 0) + 1
            
            # Track cards with exactly 1, 2, or 3 appearances in this round
            for card, count in round_card_counts.items():
                if count == 1:
                    single_hit_cards[card] += 1
                elif count == 2:
                    double_hit_cards[card] += 1
                elif count == 3:
                    triple_hit_cards[card] += 1
            
            # Store round-level data
            round_data.append({
                'round': round_num,
                'landed_cards': landed_cards,
                'hits': hits,
                'payout': payout,
                'cumulative_profit': cumulative_profit[-1]
            })
        
        # Display individual round results if verbose mode is enabled
        if verbose:
            result_text = "WIN" if payout > 0 else "LOSE"
            profit_loss = f"+${payout:.2f}" if payout > 0 else f"-${abs(payout):.2f}"
            cards_str = ", ".join(landed_cards)
            print(f"Round {round_num:5d}: [{cards_str}] | Player bet '{chosen_card}' | {hits} hits | {result_text} {profit_loss}")
    
    total_profit = sum(results)
    average_profit = total_profit / num_rounds
    win_rate = sum(1 for r in results if r > 0) / num_rounds
    
    base_results = {
        'total_trials': num_rounds,
        'total_balls_dropped': num_rounds * game_model.balls_per_round,
        'wins': sum(1 for r in results if r > 0),
        'losses': sum(1 for r in results if r <= 0),
        'win_rate': win_rate,
        'total_profit': total_profit,
        'average_return_per_play': average_profit,
        'hits_distribution': hits_distribution
    }
    
    if detailed:
        base_results['detailed_data'] = {
            'round_data': round_data,
            'cumulative_profit': cumulative_profit,
            'card_frequencies': card_frequencies,
            'single_hit_cards': single_hit_cards,
            'double_hit_cards': double_hit_cards,
            'triple_hit_cards': triple_hit_cards,
            'per_round_payouts': results
        }
    
    return base_results


def run_simulations(chosen_card: str = 'A', num_trials: int = 100000, seed: int = 42, verbose: bool = False):
    """
    Run both fair and tweaked game simulations and return comparison results.
    
    Args:
        chosen_card: Player's chosen card
        num_trials: Number of rounds per game (each round = 3 balls)
        seed: Random seed for reproducibility
        verbose: If True, display each round result (default: False)
        
    Returns:
        Tuple of (fair_results, tweaked_results, comparison_df, fair_ev, tweaked_ev)
    """
    # Initialize games
    fair_game = FairDropBall(cards, balls_per_round=BALLS_PER_ROUND)
    tweaked_game = TweakedDropBall(cards, balls_per_round=BALLS_PER_ROUND)
    
    # Calculate theoretical EVs
    fair_ev = calculate_theoretical_ev(fair_game, bet)
    tweaked_ev = calculate_theoretical_ev(tweaked_game, bet)
    
    # Run simulations
    print(f"Running {num_trials:,} rounds for each game model...")
    print(f"(Each round = {BALLS_PER_ROUND} balls, Total balls = {num_trials * BALLS_PER_ROUND:,})")
    
    if verbose:
        print("\n" + "="*70)
        print("FAIR GAME - Detailed Results")
        print("="*70)
    fair_results = PlayDropBallGame(fair_game, chosen_card, num_trials, seed, verbose)
    
    if verbose:
        print("\n" + "="*70)
        print("TWEAKED GAME - Detailed Results")
        print("="*70)
    tweaked_results = PlayDropBallGame(tweaked_game, chosen_card, num_trials, seed, verbose)
    
    # Create comparison DataFrame
    comparison = pd.DataFrame({
        'Metric': ['Total Rounds', 'Total Balls Dropped', 'Total Profit/Loss', 
                   'Average Return per Round', 'Theoretical EV per Round'],
        'Fair Game (EV ≈0)': [
            fair_results['total_trials'],
            fair_results['total_balls_dropped'],
            f"${fair_results['total_profit']:.2f}",
            f"${fair_results['average_return_per_play']:.4f}",
            f"${fair_ev['total_ev']:.4f}"
        ],
        'Tweaked Game (House Edge)': [
            tweaked_results['total_trials'],
            tweaked_results['total_balls_dropped'],
            f"${tweaked_results['total_profit']:.2f}",
            f"${tweaked_results['average_return_per_play']:.4f}",
            f"${tweaked_ev['total_ev']:.4f}"
        ]
    })
    
    return fair_results, tweaked_results, comparison, fair_ev, tweaked_ev


def run_simulations_detailed(chosen_card: str = 'A', num_trials: int = 10000, seed: int = 42):
    """
    Run both fair and tweaked game simulations with detailed data for visualizations.
    
    Args:
        chosen_card: Player's chosen card
        num_trials: Number of rounds per game (each round = 3 balls)
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary with comprehensive results for both games including detailed tracking data
    """
    # Initialize games
    fair_game = FairDropBall(cards, balls_per_round=BALLS_PER_ROUND)
    tweaked_game = TweakedDropBall(cards_with_joker, balls_per_round=BALLS_PER_ROUND)
    
    # Calculate theoretical EVs
    fair_ev = calculate_theoretical_ev(fair_game, bet)
    tweaked_ev = calculate_theoretical_ev(tweaked_game, bet)
    
    print(f"Running {num_trials:,} rounds for each game model with detailed tracking...")
    print(f"(Each round = {BALLS_PER_ROUND} balls, Total balls = {num_trials * BALLS_PER_ROUND:,})")
    
    # Run simulations with detailed tracking
    fair_results = PlayDropBallGame(fair_game, chosen_card, num_trials, seed, verbose=False, detailed=True)
    tweaked_results = PlayDropBallGame(tweaked_game, chosen_card, num_trials, seed, verbose=False, detailed=True)
    
    print(f"\n✓ Fair Game: Total Profit = ${fair_results['total_profit']:.2f}, Win Rate = {fair_results['win_rate']*100:.2f}%")
    print(f"✓ Tweaked Game: Total Profit = ${tweaked_results['total_profit']:.2f}, Win Rate = {tweaked_results['win_rate']*100:.2f}%")
    
    return {
        'fair_game': fair_game,
        'tweaked_game': tweaked_game,
        'fair_results': fair_results,
        'tweaked_results': tweaked_results,
        'fair_ev': fair_ev,
        'tweaked_ev': tweaked_ev,
        'chosen_card': chosen_card,
        'num_trials': num_trials,
        'bet': bet
    }


def print_theoretical_comparison():
    """
    Print theoretical probabilities and house edge for both Fair and Tweaked games.
    """
    print("="*70)
    print("THEORETICAL COMPARISON")
    print("="*70)
    
    # Fair Game Analysis
    print("\n--- FAIR GAME ---")
    print("\nProbabilities (per card):")
    fair_prob = 1/6
    for card in cards:
        print(f"{card}: {fair_prob*100:.2f}%")
    
    # Calculate Fair Game House Edge
    fair_game = FairDropBall(cards, balls_per_round=BALLS_PER_ROUND)
    fair_ev = calculate_theoretical_ev(fair_game, bet)
    fair_house_edge = -fair_ev['total_ev'] * 100  # Negative EV = House Edge
    print(f"\nHouse Edge: {fair_house_edge:.2f}%")
    
    # Tweaked Game Analysis
    print("\n--- TWEAKED GAME ---")
    print("\nProbabilities (per card):")
    tweaked_prob = 1/7
    for card in cards_with_joker:
        print(f"{card}: {tweaked_prob*100:.2f}%")
    
    # Calculate Tweaked Game House Edge
    tweaked_game = TweakedDropBall(cards_with_joker, balls_per_round=BALLS_PER_ROUND)
    tweaked_ev = calculate_theoretical_ev(tweaked_game, bet)
    tweaked_house_edge = -tweaked_ev['total_ev'] * 100  # Negative EV = House Edge
    print(f"\nHouse Edge: {tweaked_house_edge:.2f}%")
    
    print("\n" + "="*70)


def main():
    """
    Main execution function to run simulations and generate outputs.
    """
    # Print theoretical comparison first
    print_theoretical_comparison()
    
    # Initialize games
    fair_game = FairDropBall(cards, balls_per_round=BALLS_PER_ROUND)
    tweaked_game = TweakedDropBall(cards_with_joker, balls_per_round=BALLS_PER_ROUND)
    
    print("\n" + "="*70)
    print("SIMULATION RESULTS")
    print("="*70)
    
    # Run 3 simulations for Fair Game
    for i in range(3):
        fair_results = PlayDropBallGame(fair_game, chosenCard, NumDrop, seed=None, verbose=False)
        fair_return_pct = (fair_results['average_return_per_play'] / bet) * 100
        print(f"\n{NumDrop} drops of ball:")
        print(f"Expected return betting ${bet} = {fair_return_pct:.3f}%  # Fair Game")
    
    # Run 3 simulations for Tweaked Game
    for i in range(3):
        tweaked_results = PlayDropBallGame(tweaked_game, chosenCard, NumDrop, seed=None, verbose=False)
        tweaked_return_pct = (tweaked_results['average_return_per_play'] / bet) * 100
        print(f"\n{NumDrop} drops of ball:")
        print(f"Expected return betting ${bet} = {tweaked_return_pct:.3f}%  # Tweaked Game")


if __name__ == "__main__":
    main()