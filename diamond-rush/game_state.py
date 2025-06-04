from dataclasses import dataclass
from typing import FrozenSet

from position import Position

# Game State for A* search
@dataclass(frozen=True)
class GameState:
    """Immutable game state for A* search"""
    player_pos: Position
    has_key: bool
    rocks: FrozenSet[Position]
    diamonds: FrozenSet[Position]
    buttons_pressed: FrozenSet[Position]
    doors_open: FrozenSet[Position]
    spikes_down: FrozenSet[Position]
    holes_closed: FrozenSet[Position]
    keys: FrozenSet[Position]
    cost: int = 0

    def __post_init__(self):
        # Ensure defaults are properly set
        if self.rocks is None:
            object.__setattr__(self, 'rocks', frozenset())
        if self.diamonds is None:
            object.__setattr__(self, 'diamonds', frozenset())
        if self.buttons_pressed is None:
            object.__setattr__(self, 'buttons_pressed', frozenset())
        if self.doors_open is None:
            object.__setattr__(self, 'doors_open', frozenset())
        if self.spikes_down is None:
            object.__setattr__(self, 'spikes_down', frozenset())
        if self.holes_closed is None:
            object.__setattr__(self, 'holes_closed', frozenset())
        if self.keys is None:
            object.__setattr__(self, 'keys', frozenset())
        if self.has_key is None:
            object.__setattr__(self, 'has_key', False)

    def __hash__(self):
        return hash((self.player_pos, self.rocks, self.diamonds,
                     self.buttons_pressed, self.doors_open, self.keys, self.has_key, self.spikes_down, self.holes_closed))

    def __eq__(self, other):
        return (self.player_pos == other.player_pos and
                self.rocks == other.rocks and
                self.diamonds == other.diamonds and
                self.buttons_pressed == other.buttons_pressed and
                self.doors_open == other.doors_open and
                self.keys == other.keys and
                self.has_key == other.has_key and
                self.spikes_down == other.spikes_down and
                self.holes_closed == other.holes_closed
                )