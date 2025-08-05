from typing import Dict, Tuple
from models import Decision, Rank

# Decision shortcuts for readability
H = Decision.HIT
S = Decision.STAND
D = Decision.DOUBLE
SP = Decision.SPLIT


def get_hard_totals_table() -> Dict[Tuple[int, int], Decision]:
    """
    Hard totals strategy table.
    Key: (player_total, dealer_value) → Decision
    """
    return {
        # Total 9
        (9, 2): H,
        (9, 3): D,
        (9, 4): D,
        (9, 5): D,
        (9, 6): D,
        (9, 7): H,
        (9, 8): H,
        (9, 9): H,
        (9, 10): H,
        (9, 11): H,
        # Total 10
        (10, 2): D,
        (10, 3): D,
        (10, 4): D,
        (10, 5): D,
        (10, 6): D,
        (10, 7): D,
        (10, 8): D,
        (10, 9): D,
        (10, 10): H,
        (10, 11): H,
        # Total 11
        (11, 2): D,
        (11, 3): D,
        (11, 4): D,
        (11, 5): D,
        (11, 6): D,
        (11, 7): D,
        (11, 8): D,
        (11, 9): D,
        (11, 10): D,
        (11, 11): D,
        # Total 12
        (12, 2): H,
        (12, 3): H,
        (12, 4): S,
        (12, 5): S,
        (12, 6): S,
        (12, 7): H,
        (12, 8): H,
        (12, 9): H,
        (12, 10): H,
        (12, 11): H,
        # Total 13
        (13, 2): S,
        (13, 3): S,
        (13, 4): S,
        (13, 5): S,
        (13, 6): S,
        (13, 7): H,
        (13, 8): H,
        (13, 9): H,
        (13, 10): H,
        (13, 11): H,
        # Total 14
        (14, 2): S,
        (14, 3): S,
        (14, 4): S,
        (14, 5): S,
        (14, 6): S,
        (14, 7): H,
        (14, 8): H,
        (14, 9): H,
        (14, 10): H,
        (14, 11): H,
        # Total 15
        (15, 2): S,
        (15, 3): S,
        (15, 4): S,
        (15, 5): S,
        (15, 6): S,
        (15, 7): H,
        (15, 8): H,
        (15, 9): H,
        (15, 10): H,
        (15, 11): H,
        # Total 16
        (16, 2): S,
        (16, 3): S,
        (16, 4): S,
        (16, 5): S,
        (16, 6): S,
        (16, 7): H,
        (16, 8): H,
        (16, 9): H,
        (16, 10): H,
        (16, 11): H,
    }


def get_soft_totals_table() -> Dict[Tuple[int, int], Decision]:
    """
    Soft totals strategy table.
    Key: (player_total, dealer_value) → Decision
    """
    return {
        # Soft 13-14 (A,2 or A,3)
        (13, 2): H,
        (13, 3): H,
        (13, 4): H,
        (13, 5): D,
        (13, 6): D,
        (13, 7): H,
        (13, 8): H,
        (13, 9): H,
        (13, 10): H,
        (13, 11): H,
        (14, 2): H,
        (14, 3): H,
        (14, 4): H,
        (14, 5): D,
        (14, 6): D,
        (14, 7): H,
        (14, 8): H,
        (14, 9): H,
        (14, 10): H,
        (14, 11): H,
        # Soft 15-16
        (15, 2): H,
        (15, 3): H,
        (15, 4): D,
        (15, 5): D,
        (15, 6): D,
        (15, 7): H,
        (15, 8): H,
        (15, 9): H,
        (15, 10): H,
        (15, 11): H,
        (16, 2): H,
        (16, 3): H,
        (16, 4): D,
        (16, 5): D,
        (16, 6): D,
        (16, 7): H,
        (16, 8): H,
        (16, 9): H,
        (16, 10): H,
        (16, 11): H,
        # Soft 17
        (17, 2): H,
        (17, 3): D,
        (17, 4): D,
        (17, 5): D,
        (17, 6): D,
        (17, 7): H,
        (17, 8): H,
        (17, 9): H,
        (17, 10): H,
        (17, 11): H,
        # Soft 18
        (18, 2): S,
        (18, 3): D,
        (18, 4): D,
        (18, 5): D,
        (18, 6): D,
        (18, 7): S,
        (18, 8): S,
        (18, 9): H,
        (18, 10): H,
        (18, 11): H,
    }


def get_pair_splits_table() -> Dict[Tuple[Rank, int], Decision]:
    """
    Pair splitting strategy table.
    Key: (rank, dealer_value) → Decision (SPLIT or HIT)
    """
    return {
        # Aces - always split
        (Rank.ACE, 2): SP,
        (Rank.ACE, 3): SP,
        (Rank.ACE, 4): SP,
        (Rank.ACE, 5): SP,
        (Rank.ACE, 6): SP,
        (Rank.ACE, 7): SP,
        (Rank.ACE, 8): SP,
        (Rank.ACE, 9): SP,
        (Rank.ACE, 10): SP,
        (Rank.ACE, 11): SP,
        # 8s - always split
        (Rank.EIGHT, 2): SP,
        (Rank.EIGHT, 3): SP,
        (Rank.EIGHT, 4): SP,
        (Rank.EIGHT, 5): SP,
        (Rank.EIGHT, 6): SP,
        (Rank.EIGHT, 7): SP,
        (Rank.EIGHT, 8): SP,
        (Rank.EIGHT, 9): SP,
        (Rank.EIGHT, 10): SP,
        (Rank.EIGHT, 11): SP,
        # 9s - split except vs 7, 10, A
        (Rank.NINE, 2): SP,
        (Rank.NINE, 3): SP,
        (Rank.NINE, 4): SP,
        (Rank.NINE, 5): SP,
        (Rank.NINE, 6): SP,
        (Rank.NINE, 7): S,
        (Rank.NINE, 8): SP,
        (Rank.NINE, 9): SP,
        (Rank.NINE, 10): S,
        (Rank.NINE, 11): S,
        # 7s - split vs 2-7
        (Rank.SEVEN, 2): SP,
        (Rank.SEVEN, 3): SP,
        (Rank.SEVEN, 4): SP,
        (Rank.SEVEN, 5): SP,
        (Rank.SEVEN, 6): SP,
        (Rank.SEVEN, 7): SP,
        (Rank.SEVEN, 8): H,
        (Rank.SEVEN, 9): H,
        (Rank.SEVEN, 10): H,
        (Rank.SEVEN, 11): H,
        # 6s - split vs 2-6
        (Rank.SIX, 2): SP,
        (Rank.SIX, 3): SP,
        (Rank.SIX, 4): SP,
        (Rank.SIX, 5): SP,
        (Rank.SIX, 6): SP,
        (Rank.SIX, 7): H,
        (Rank.SIX, 8): H,
        (Rank.SIX, 9): H,
        (Rank.SIX, 10): H,
        (Rank.SIX, 11): H,
        # 4s - split vs 5-6 only
        (Rank.FOUR, 2): H,
        (Rank.FOUR, 3): H,
        (Rank.FOUR, 4): H,
        (Rank.FOUR, 5): SP,
        (Rank.FOUR, 6): SP,
        (Rank.FOUR, 7): H,
        (Rank.FOUR, 8): H,
        (Rank.FOUR, 9): H,
        (Rank.FOUR, 10): H,
        (Rank.FOUR, 11): H,
        # 3s and 2s - split vs 2-7
        (Rank.THREE, 2): SP,
        (Rank.THREE, 3): SP,
        (Rank.THREE, 4): SP,
        (Rank.THREE, 5): SP,
        (Rank.THREE, 6): SP,
        (Rank.THREE, 7): SP,
        (Rank.THREE, 8): H,
        (Rank.THREE, 9): H,
        (Rank.THREE, 10): H,
        (Rank.THREE, 11): H,
        (Rank.TWO, 2): SP,
        (Rank.TWO, 3): SP,
        (Rank.TWO, 4): SP,
        (Rank.TWO, 5): SP,
        (Rank.TWO, 6): SP,
        (Rank.TWO, 7): SP,
        (Rank.TWO, 8): H,
        (Rank.TWO, 9): H,
        (Rank.TWO, 10): H,
        (Rank.TWO, 11): H,
        # 10s, 5s - never split (not in table means don't split)
    }
