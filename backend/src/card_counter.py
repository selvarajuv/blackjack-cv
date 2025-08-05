from typing import List, Dict
from models import Card, Rank


class CardCounter:
    """
    Implements various card counting systems.
    Default is Hi-Lo, the most popular and balanced system.
    """

    def __init__(self, num_decks: int = 6, system: str = "hi-lo"):
        self.num_decks = num_decks
        self.cards_seen = 0
        self.running_count = 0
        self.system = system

        # Define counting systems
        self.count_values = self._get_count_system(system)

    def _get_count_system(self, system: str) -> Dict[Rank, int]:
        """Get card values for different counting systems"""

        if system == "hi-lo":
            # Hi-Lo: Most popular balanced system
            return {
                Rank.TWO: 1,
                Rank.THREE: 1,
                Rank.FOUR: 1,
                Rank.FIVE: 1,
                Rank.SIX: 1,  # Low cards: +1
                Rank.SEVEN: 0,
                Rank.EIGHT: 0,
                Rank.NINE: 0,  # Neutral: 0
                Rank.TEN: -1,
                Rank.JACK: -1,
                Rank.QUEEN: -1,
                Rank.KING: -1,
                Rank.ACE: -1,  # High cards: -1
            }

        elif system == "ko":
            # Knock-Out: Easier unbalanced system (no true count conversion)
            return {
                Rank.TWO: 1,
                Rank.THREE: 1,
                Rank.FOUR: 1,
                Rank.FIVE: 1,
                Rank.SIX: 1,
                Rank.SEVEN: 1,  # Seven is +1 in KO
                Rank.EIGHT: 0,
                Rank.NINE: 0,
                Rank.TEN: -1,
                Rank.JACK: -1,
                Rank.QUEEN: -1,
                Rank.KING: -1,
                Rank.ACE: -1,
            }

        elif system == "omega-ii":
            # Omega II: More complex but more accurate
            return {
                Rank.TWO: 1,
                Rank.THREE: 1,
                Rank.FOUR: 2,
                Rank.FIVE: 2,
                Rank.SIX: 2,
                Rank.SEVEN: 1,
                Rank.EIGHT: 0,
                Rank.NINE: -1,
                Rank.TEN: -2,
                Rank.JACK: -2,
                Rank.QUEEN: -2,
                Rank.KING: -2,
                Rank.ACE: 0,
            }

        else:
            raise ValueError(f"Unknown counting system: {system}")

    def update_count(self, cards: List[Card]) -> None:
        """Update running count based on new cards seen"""
        for card in cards:
            count_value = self.count_values.get(card.rank, 0)
            self.running_count += count_value
            self.cards_seen += 1

    def get_true_count(self) -> float:
        """
        Calculate true count (running count / decks remaining).
        More accurate than running count for betting decisions.
        """
        # Estimate decks remaining
        cards_per_deck = 52
        total_cards = self.num_decks * cards_per_deck
        cards_remaining = total_cards - self.cards_seen
        decks_remaining = cards_remaining / cards_per_deck

        # Avoid division by zero
        if decks_remaining < 0.5:
            decks_remaining = 0.5

        return self.running_count / decks_remaining

    def get_bet_multiplier(self) -> float:
        """
        Suggest bet sizing based on true count.
        Basic strategy: Increase bets when count is positive.
        """
        true_count = self.get_true_count()

        if true_count <= 0:
            return 1.0  # Minimum bet
        elif true_count <= 2:
            return 2.0  # 2x bet
        elif true_count <= 3:
            return 3.0  # 3x bet
        elif true_count <= 4:
            return 4.0  # 4x bet
        else:
            return 5.0  # Max bet (to avoid suspicion)

    def should_take_insurance(self) -> bool:
        """Insurance is profitable when true count >= 3 (Hi-Lo system)"""
        return self.get_true_count() >= 3.0

    def get_strategy_deviations(
        self, player_total: int, dealer_card: int, is_soft: bool = False
    ) -> Dict[str, any]:
        """
        Return strategy adjustments based on count.
        These are called "Index plays" or "Deviations".
        """
        true_count = self.get_true_count()
        deviations = {}

        # Example deviations for Hi-Lo system
        # Format: (player_total, dealer_card, is_soft) -> (action_if_count_>=_threshold, threshold)

        # 16 vs 10: Stand if true count >= 0 (instead of always hit)
        if player_total == 16 and dealer_card == 10 and not is_soft:
            if true_count >= 0:
                deviations["16v10"] = "stand"

        # 15 vs 10: Stand if true count >= 4
        if player_total == 15 and dealer_card == 10 and not is_soft:
            if true_count >= 4:
                deviations["15v10"] = "stand"

        # 12 vs 3: Stand if true count >= 2 (instead of hit)
        if player_total == 12 and dealer_card == 3 and not is_soft:
            if true_count >= 2:
                deviations["12v3"] = "stand"

        # 12 vs 2: Stand if true count >= 3
        if player_total == 12 and dealer_card == 2 and not is_soft:
            if true_count >= 3:
                deviations["12v2"] = "stand"

        # Double 9 vs 2: Double if true count >= 1
        if player_total == 9 and dealer_card == 2 and not is_soft:
            if true_count >= 1:
                deviations["9v2"] = "double"

        return deviations

    def reset(self) -> None:
        """Reset count (new shoe)"""
        self.running_count = 0
        self.cards_seen = 0

    def get_count_stats(self) -> Dict[str, any]:
        """Get current counting statistics"""
        return {
            "system": self.system,
            "running_count": self.running_count,
            "true_count": round(self.get_true_count(), 2),
            "cards_seen": self.cards_seen,
            "decks_remaining": round((self.num_decks * 52 - self.cards_seen) / 52, 2),
            "bet_multiplier": self.get_bet_multiplier(),
            "take_insurance": self.should_take_insurance(),
        }
