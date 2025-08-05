from pydantic import BaseModel
from typing import List, Optional
from enum import Enum


class Suit(str, Enum):
    HEARTS = "hearts"
    DIAMONDS = "diamonds"
    CLUBS = "clubs"
    SPADES = "spades"


class Rank(str, Enum):
    ACE = "A"
    TWO = "2"
    THREE = "3"
    FOUR = "4"
    FIVE = "5"
    SIX = "6"
    SEVEN = "7"
    EIGHT = "8"
    NINE = "9"
    TEN = "10"
    JACK = "J"
    QUEEN = "Q"
    KING = "K"


class Card(BaseModel):
    rank: Rank
    suit: Suit

    @property
    def value(self) -> int:
        """Returns blackjack value of the card"""
        if self.rank in [Rank.JACK, Rank.QUEEN, Rank.KING]:
            return 10
        elif self.rank == Rank.ACE:
            return 11  # Will handle soft/hard aces in Hand class
        else:
            return int(self.rank.value)


class Hand(BaseModel):
    cards: List[Card]

    @property
    def total(self) -> int:
        """Calculate hand total, handling soft/hard aces"""
        total = sum(card.value for card in self.cards)
        aces = sum(1 for card in self.cards if card.rank == Rank.ACE)

        # Convert aces from 11 to 1 if needed
        while total > 21 and aces > 0:
            total -= 10
            aces -= 1

        return total

    @property
    def is_soft(self) -> bool:
        """Check if hand has a soft ace (ace counted as 11)"""
        total = sum(card.value for card in self.cards)
        aces = sum(1 for card in self.cards if card.rank == Rank.ACE)

        return aces > 0 and total <= 21

    @property
    def is_blackjack(self) -> bool:
        """Check if hand is a natural blackjack"""
        return len(self.cards) == 2 and self.total == 21

    @property
    def is_bust(self) -> bool:
        """Check if hand is bust (over 21)"""
        return self.total > 21


class Decision(str, Enum):
    HIT = "hit"
    STAND = "stand"
    DOUBLE = "double"
    SPLIT = "split"
    SURRENDER = "surrender"


class GameState(BaseModel):
    player_hand: Hand
    dealer_card: Card
    recommended_action: Optional[Decision] = None
    true_count: float = 0.0
    running_count: int = 0


class DetectedCard(BaseModel):
    """Card detected by computer vision"""

    card: Card
    confidence: float
    bbox: List[int]  # [x, y, width, height]
