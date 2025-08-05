from typing import Optional
from .models import Hand, Card, Decision, Rank
from .strategy_tables import (
    get_hard_totals_table,
    get_soft_totals_table,
    get_pair_splits_table,
)


class BasicStrategy:
    """
    Implements basic blackjack strategy based on statistical optimal play.
    Does not consider card counting - just mathematically best decisions.
    """

    def __init__(self):
        # Load strategy tables from separate file
        self._hard_totals = get_hard_totals_table()
        self._soft_totals = get_soft_totals_table()
        self._pair_splits = get_pair_splits_table()

    def get_decision(
        self,
        player_hand: Hand,
        dealer_card: Card,
        can_double: bool = True,
        can_split: bool = True,
    ) -> Decision:
        """
        Get the statistically optimal decision for a given hand.

        Args:
            player_hand: The player's current hand
            dealer_card: The dealer's up card
            can_double: Whether doubling is allowed (usually only on first 2 cards)
            can_split: Whether splitting is allowed
        """
        dealer_value = self._get_dealer_value(dealer_card)

        # Check for pairs first
        if self._is_pair(player_hand) and can_split:
            split_decision = self._check_pair_split(
                player_hand.cards[0].rank, dealer_value
            )
            if split_decision == Decision.SPLIT:
                return split_decision
            # If not splitting the pair, continue to normal hand evaluation

        # Check soft hands (with Ace counted as 11)
        if player_hand.is_soft and player_hand.total <= 21:
            decision = self._check_soft_total(player_hand.total, dealer_value)
        else:
            # Hard totals
            decision = self._check_hard_total(player_hand.total, dealer_value)

        # Adjust for game rules
        if decision == Decision.DOUBLE and not can_double:
            # If can't double, hit instead (except on soft 18 vs 2-6, then stand)
            if player_hand.is_soft and player_hand.total == 18 and dealer_value <= 6:
                return Decision.STAND
            return Decision.HIT

        return decision

    def _get_dealer_value(self, card: Card) -> int:
        """Get dealer card value for strategy lookup"""
        if card.rank == Rank.ACE:
            return 11
        return card.value

    def _is_pair(self, hand: Hand) -> bool:
        """Check if hand is a pair"""
        if len(hand.cards) != 2:
            return False
        return hand.cards[0].rank == hand.cards[1].rank

    def _check_hard_total(self, total: int, dealer_value: int) -> Decision:
        """Look up decision for hard totals"""
        if total >= 17:
            return Decision.STAND
        if total <= 8:
            return Decision.HIT

        return self._hard_totals.get((total, dealer_value), Decision.HIT)

    def _check_soft_total(self, total: int, dealer_value: int) -> Decision:
        """Look up decision for soft totals"""
        if total >= 19:
            return Decision.STAND
        if total <= 13:
            return Decision.HIT

        return self._soft_totals.get((total, dealer_value), Decision.HIT)

    def _check_pair_split(self, rank: Rank, dealer_value: int) -> Optional[Decision]:
        """
        Look up decision for pairs.
        Returns SPLIT if should split, None otherwise.
        """
        return self._pair_splits.get((rank, dealer_value), None)
