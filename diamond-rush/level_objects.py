from enum import Enum

# Display characters for console visualization
class CellDisplay:
    WALL = '#'
    PLAYER = 'P'
    ROCK = 'R'
    BUTTON = 'B'
    DOOR_CLOSED = 'D'
    DOOR_OPEN = 'd'
    DIAMOND = '*'
    KEY = 'K'
    KEY_DOOR = 'L'
    LAVA = 'X'
    SPIKES_UP = '^'
    SPIKES_DOWN = 'v'
    EMPTY = '.'
    EXIT_CLOSED = 'E'
    EXIT_OPEN = 'e'
    HOLE = 'O'

class LevelObjects(Enum):
    DIAMOND = "diamond.png"
    BUTTON = "button.png"
    DOOR_CLOSED = "door_closed.png"
    DOOR_OPEN = "door_open.png"
    EXIT_CLOSED = "exit_closed.png"
    EXIT_OPEN = "exit_open.png"
    KEY = "key.png"
    KEY_DOOR = "key_door.png"
    LAVA = "lava.png"
    PLAYER_LEFT = "player_left.png"
    PLAYER_RIGHT = "player_right.png"
    ROCK = "rock.png"
    SPIKES_DOWN = "spikes_down.png"
    SPIKES_UP = "spikes_up.png"
    WALL = "wall.png"
    HOLE = "hole.png"
    EMPTY = "empty.png"
