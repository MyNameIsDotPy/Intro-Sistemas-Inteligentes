from game_controller import GameController
from screen_capture import ScreenCapture
import time
from game_solver import DiamondRushSolver
from utils import print_game_state, convert_to_solver_state
from game_vision import GameVision

# Main automation class
class DiamondRushBot:
    """Main class that coordinates all components"""

    def __init__(self, template_folder='templates/'):
        self.vision = GameVision(template_folder)
        self.controller = GameController()
        self.screen_capture = ScreenCapture()
        self.solver = None

        print("\nDiamond Rush Bot Initialized")
        print("=" * 50)
        print("Components: Vision, Controller, Screen Capture, A* Solver")

    def solve_current_level(self):
        """Main method to solve the current level"""
        try:
            print("\n" + "=" * 50)
            print("STARTING LEVEL SOLVE SEQUENCE")
            print("=" * 50)

            # Capture and analyze game state
            screenshot = self.screen_capture.capture_game_area()
            game_state = self.vision.get_game_state(screenshot)

            print("\nCurrent game state:")
            print_game_state(game_state)

            # Set up solver and find solution
            initial_state = convert_to_solver_state(game_state)
            self.solver = DiamondRushSolver(initial_state, game_state)
            solution_path = self.solver.solve()

            if solution_path:
                print(f"\nSolution found with {len(solution_path)} steps!")

                # Execute solution
                success = self.controller.execute_action_sequence(solution_path)
                return success
            else:
                print("\nNo solution found!")
                return False

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error: {e}")
            print(f"Error: {e.__traceback__}")
            return False

    def run_continuous(self, max_levels=10):
        """Solve multiple levels continuously"""
        print(f"\nStarting continuous mode (max {max_levels} levels)")

        levels_completed = 0

        for level in range(1, max_levels + 1):
            print(f"\n{'=' * 20} LEVEL {level} {'=' * 20}")

            success = self.solve_current_level()

            if success:
                levels_completed += 1
                print(f"Level {level} completed!")
                time.sleep(2)
            else:
                print(f"Failed to complete level {level}")
                break

        print(f"\nCompleted {levels_completed}/{max_levels} levels")
        return levels_completed


# Main entry point
if __name__ == "__main__":
    time.sleep(3)
    bot = DiamondRushBot()
    bot.solve_current_level()
