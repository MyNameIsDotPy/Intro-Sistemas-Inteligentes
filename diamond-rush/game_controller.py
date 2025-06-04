import time
import pyautogui
from position import Position

# Game Controller
class GameController:
    """Controls the game through simulated keyboard input"""

    def __init__(self, key_mapping=None):
        if key_mapping is None:
            key_mapping = {
                'up': 'up',
                'down': 'down',
                'left': 'left',
                'right': 'right',
                'action': 'space'
            }

        self.key_mapping = key_mapping
        self.move_delay = 0.1  # Delay between moves (adjust based on game speed)
        print("Game controller initialized")

    def execute_move(self, direction):
        """Execute a single move in the game"""
        direction_map = {
            Position(0, -1): 'up',
            Position(1, 0): 'right',
            Position(0, 1): 'down',
            Position(-1, 0): 'left'
        }

        if direction in direction_map:
            key = self.key_mapping[direction_map[direction]]
            print(f"Pressing key: {key}")
            pyautogui.keyDown(key)
            time.sleep(self.move_delay)
            pyautogui.keyUp(key)
            time.sleep(self.move_delay * 3)
            return True

        return False

    def execute_action_sequence(self, path):
        """Execute a sequence of moves from A* solution"""
        if not path or len(path) < 2:
            print("No valid path to execute")
            return False

        print(f"Executing {len(path) - 1} moves...")

        for i in range(len(path) - 1):
            current_state = path[i]
            next_state = path[i + 1]

            # Calculate direction
            direction = next_state.player_pos - current_state.player_pos

            print(f"Move {i + 1}: {current_state.player_pos} -> {next_state.player_pos}")

            # Check for rock push
            if next_state.cost - current_state.cost > 1:
                print("  (Rock push)")

            success = self.execute_move(direction)
            if not success:
                print(f"Failed at move {i + 1}")
                return False


        print("Action sequence executed successfully")
        return True
