import cv2
import pyautogui
import time
import heapq
from enum import Enum
from typing import Tuple, List, Set, Dict, Optional, FrozenSet
from dataclasses import dataclass
from utils import detectar_piedras_redondas
from initial_config import *
from memory import load_button_door_map, save_button_door_map
from common import Position

# Game Objects Enum as provided
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



# Movement directions
DIRECTIONS = [
    Position(0, -1),  # UP
    Position(1, 0),  # RIGHT
    Position(0, 1),  # DOWN
    Position(-1, 0),  # LEFT
]


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


# Computer Vision Module
class GameVision:
    """Handles game state detection using computer vision"""

    def __init__(self, template_folder='templates/'):
        self.template_folder = template_folder
        self.templates = self._load_templates()
        print(f"Vision system initialized with template folder: {template_folder}")

    def _load_templates(self):
        """Load all game object templates"""
        templates = {}
        for obj in LevelObjects:
            template_path = f"{self.template_folder}/{obj.value}"
            templates[obj] = cv2.imread(template_path)
            if templates[obj] is None:
                print(f"Warning: Could not load template {template_path}")
        return templates

    def find_all_templates(self, screen, template, threshold=0.8):
        """Find all occurrences of a template in the screen using multi-template matching"""
        if template is None:
            return []

        result = cv2.matchTemplate(screen, template, cv2.TM_CCOEFF_NORMED)
        locations = np.where(result >= threshold)
        h, w = template.shape[:2]

        matches = []
        for pt in zip(*locations[::-1]):
            matches.append((pt[0], pt[1], w, h))

        # Apply non-maximum suppression
        return self._non_max_suppression(matches)

    @staticmethod
    def _non_max_suppression(boxes, overlap_thresh=0.3):
        """Apply non-maximum suppression to remove overlapping detections"""
        if len(boxes) == 0:
            return []

        # Convert to numpy array for processing
        boxes = np.array(boxes)

        # Extract coordinates
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 0] + boxes[:, 2]
        y2 = boxes[:, 1] + boxes[:, 3]

        # Calculate areas
        area = (x2 - x1 + 1) * (y2 - y1 + 1)

        # Sort by y2 coordinate
        idxs = np.argsort(y2)

        pick = []
        while len(idxs) > 0:
            # Pick the last index
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            # Find overlapping boxes
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            # Calculate overlap area
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            overlap = (w * h) / area[idxs[:last]]

            # Remove indices with significant overlap
            idxs = np.delete(idxs, np.concatenate(([last],
                                                   np.where(overlap > overlap_thresh)[0])))

        return boxes[pick].astype("int").tolist()

    def get_game_state(self, screenshot):
        """Process screenshot to extract game state"""
        # Convert to grayscale for better template matching

        # Create empty grid based on game dimensions
        # (For actual implementation, detect grid dimensions first)
        rows, cols = GRID_DIMENSIONS
        cell_size = screenshot.shape[1] // cols

        grid = [[LevelObjects.EMPTY for _ in range(cols)] for _ in range(rows)]

        # Track object positions
        player_pos = None
        diamonds = []
        rocks = []
        holes = []
        spikes = {}
        buttons = []
        doors = {}
        exit_pos = None
        exit_open = False
        keys = []
        key_doors = []

        for y in range(rows):
            for x in range(cols):
                a_x, a_y = int(x * SINGLE_GRID_SIZE[1]), int(y * SINGLE_GRID_SIZE[0])
                b_x, b_y = a_x + int(SINGLE_GRID_SIZE[1]), a_y + int(SINGLE_GRID_SIZE[0])

                crop = screenshot[a_y:b_y, a_x:b_x]
                color_mean_bgr = cv2.mean(crop)[:3]

                diff = np.linalg.norm(color_mean_bgr - GROUND_COLOR_BGR)

                if diff > 50:
                    grid[y][x] = LevelObjects.WALL

        # Detect all rocks
        piedras = detectar_piedras_redondas(screenshot, mostrar_resultado=False)

        for piedra in piedras:
            x, y, r = piedra
            x = x//cell_size
            y = y//cell_size

            if x >= cols or y >= rows or r < 10:
                continue
            rocks.append(Position(x, y))
            grid[y][x] = LevelObjects.ROCK

        # Detect all game objects
        for obj, template in self.templates.items():
            if template is None:
                continue

            matches = self.find_all_templates(screenshot, template)

            for (x, y, w, h) in matches:
                # Convert pixel coordinates to grid coordinates
                grid_x = x // cell_size
                grid_y = y // cell_size

                if grid_x >= cols or grid_y >= rows or obj == LevelObjects.ROCK:
                    continue

                # Update grid and object lists
                grid[grid_y][grid_x] = obj

                # Update specific object lists
                if obj in [LevelObjects.PLAYER_LEFT, LevelObjects.PLAYER_RIGHT]:
                    player_pos = Position(grid_x, grid_y)
                elif obj == LevelObjects.DIAMOND:
                    diamonds.append(Position(grid_x, grid_y))
                elif obj == LevelObjects.BUTTON:
                    buttons.append(Position(grid_x, grid_y))
                elif obj == LevelObjects.DOOR_CLOSED:
                    doors[Position(grid_x, grid_y)] = False
                elif obj == LevelObjects.DOOR_OPEN:
                    doors[Position(grid_x, grid_y)] = True
                elif obj == LevelObjects.EXIT_CLOSED:
                    exit_pos = Position(grid_x, grid_y)
                    exit_open = False
                elif obj == LevelObjects.EXIT_OPEN:
                    exit_pos = Position(grid_x, grid_y)
                    exit_open = True
                elif obj == LevelObjects.KEY:
                    keys.append(Position(grid_x, grid_y))
                elif obj == LevelObjects.KEY_DOOR:
                    key_doors.append(Position(grid_x, grid_y))
                elif obj == LevelObjects.SPIKES_DOWN:
                    spike_pos = Position(grid_x, grid_y)
                    spikes[spike_pos] = False
                elif obj == LevelObjects.SPIKES_UP:
                    spike_pos = Position(grid_x, grid_y)
                    spikes[spike_pos] = True
                elif obj == LevelObjects.HOLE:
                    holes.append(Position(grid_x, grid_y))

        # Return structured game state
        return {
            'grid': grid,
            'player_pos': player_pos,
            'spikes': spikes,
            'diamonds': diamonds,
            'rocks': rocks,
            'buttons': buttons,
            'doors': doors,
            'exit': exit_pos,
            'exit_open': exit_open,
            'keys': keys,
            'key_doors': key_doors
        }


# A* Solver
class DiamondRushSolver:
    """A* solver for Diamond Rush puzzles"""

    def __init__(self, initial_state, grid_info):
        self.initial_state = initial_state
        self.grid = grid_info['grid']
        self.rows = len(self.grid)
        self.cols = len(self.grid[0])
        self.all_diamonds = set(grid_info['diamonds'])
        self.exit_pos = grid_info['exit']
        self.all_buttons = set(grid_info['buttons'])
        self.all_doors = set(grid_info['doors'].keys())
        self.button_door_map = self._discover_button_door_mapping()
        self.counter = 0  # For unique heap ordering

        print(f"Solver initialized for {self.rows}x{self.cols} grid")
        print(f"Goal: Collect {len(self.all_diamonds)} diamonds and reach exit at {self.exit_pos}")

    def _discover_button_door_mapping(self):
        level_id = f"{self.rows}x{self.cols}_exit_{self.exit_pos}"
        memory = load_button_door_map(level_id)
        if memory:
            print(f"Usando mapeo fijo desde memoria para {level_id}")
            return memory

        # Inicializar vacío para aprender durante el juego
        print(f"No hay memoria previa para {level_id}, aprendiendo dinámicamente")
        return {}

    def is_valid_position(self, pos):
        """Check if position is within grid bounds"""
        return 0 <= pos.x < self.cols and 0 <= pos.y < self.rows

    def is_passable(self, pos, state: GameState, is_rock: bool = False):
        """Check if position can be moved to"""
        if not self.is_valid_position(pos):
            return False

        cell = self.grid[pos.y][pos.x]

        # Impassable objects
        if cell in [LevelObjects.WALL]:
            return False

        if cell is LevelObjects.LAVA and not is_rock:
            return False

        # Check closed holes
        if cell is LevelObjects.HOLE:
            if pos not in state.holes_closed and not is_rock:
                return False

        if cell is LevelObjects.EXIT_CLOSED and state.diamonds:
            return False

        # Door check
        if cell == LevelObjects.DOOR_CLOSED and pos not in state.doors_open:
            return False

        # Spikes check
        if cell == LevelObjects.SPIKES_DOWN and pos not in state.spikes_down:
            return False

        # Key door check
        if cell == LevelObjects.KEY_DOOR:
            if pos not in state.doors_open:
                if is_rock:
                    return False
                if not state.has_key:
                    return False

        # Rock check
        if pos in state.rocks and pos not in state.holes_closed:
            return False

        return True

    def get_neighbors(self, state):
        """Generate all valid neighbor states"""
        neighbors = []
        current_pos = state.player_pos

        for direction in DIRECTIONS:
            new_pos = current_pos + direction

            # Regular movement
            if self.is_passable(new_pos, state):
                new_state = self._create_new_state(state, new_pos, current_pos)
                neighbors.append((new_state, 1))

            # Rock pushing
            elif new_pos in state.rocks:
                rock_new_pos = new_pos + direction
                if self.is_passable(rock_new_pos, state, is_rock=True):
                    new_state = self._create_new_state_with_rock_push(
                        state, new_pos, new_pos, rock_new_pos)
                    neighbors.append((new_state, 2))

        return neighbors

    def _create_new_state(self, state: GameState, new_pos, current_pose):
        """Create new state after player movement"""
        # Collect diamonds
        new_diamonds = state.diamonds
        if new_pos in new_diamonds:
            new_diamonds = new_diamonds - {new_pos}

        # Collect keys
        new_keys = state.keys
        cell = self.grid[new_pos.y][new_pos.x]
        new_has_key = state.has_key
        if cell == LevelObjects.KEY and not new_has_key and new_pos in state.keys:
            new_keys = new_keys - {new_pos}
            new_has_key = True

        # Spikes
        new_spikes = state.spikes_down
        if current_pose in new_spikes:
            new_spikes = new_spikes - {current_pose}

        # Press buttons
        new_buttons_pressed = state.buttons_pressed
        if new_pos in self.all_buttons:
            new_buttons_pressed = new_buttons_pressed | {new_pos}

        # Update doors
        new_doors_open = self._update_doors(new_buttons_pressed, state.rocks)
        # Detectar nuevas puertas abiertas
        nuevas_puertas = new_doors_open - state.doors_open
        if nuevas_puertas:
            for puerta in nuevas_puertas:
                # Asociar cada nueva puerta abierta al botón que causó el cambio
                for btn in new_buttons_pressed:
                    if btn in self.button_door_map:
                        self.button_door_map[btn].add(puerta)
                    else:
                        self.button_door_map[btn] = {puerta}

        if cell == LevelObjects.KEY_DOOR and state.has_key and new_pos not in new_doors_open:
            new_doors_open = new_doors_open | {new_pos}
            new_has_key = False

        return GameState(
            player_pos=new_pos,
            rocks=state.rocks,
            diamonds=new_diamonds,
            buttons_pressed=new_buttons_pressed,
            doors_open=new_doors_open,
            keys=new_keys,
            cost=state.cost + 1,
            spikes_down=new_spikes,
            has_key=new_has_key,
            holes_closed=state.holes_closed
        )

    def _create_new_state_with_rock_push(self, state: GameState, new_pos, rock_old_pos, rock_new_pos):
        """Create new state after pushing a rock"""

        rock_cell = self.grid[rock_new_pos.y][rock_new_pos.x]
        new_holes_closed = state.holes_closed

        if rock_cell is LevelObjects.HOLE and  rock_new_pos not in new_holes_closed:
            new_holes_closed = new_holes_closed | {rock_new_pos}

        # Update rock positions
        new_rocks = state.rocks - {rock_old_pos} | {rock_new_pos}

        # Check if rock pressed a button
        new_buttons_pressed = state.buttons_pressed
        if rock_new_pos in self.all_buttons:
            new_buttons_pressed = new_buttons_pressed | {rock_new_pos}

        # Check for diamonds
        new_diamonds = state.diamonds
        if new_pos in new_diamonds:
            new_diamonds = new_diamonds - {new_pos}

        # Update doors
        new_doors_open = self._update_doors(new_buttons_pressed, new_rocks)
        # Detectar nuevas puertas abiertas
        nuevas_puertas = new_doors_open - state.doors_open
        if nuevas_puertas:
            for puerta in nuevas_puertas:
                # Asociar cada nueva puerta abierta al botón que causó el cambio
                for btn in new_buttons_pressed:
                    if btn in self.button_door_map:
                        self.button_door_map[btn].add(puerta)
                    else:
                        self.button_door_map[btn] = {puerta}

        return GameState(
            player_pos=new_pos,
            rocks=new_rocks,
            diamonds=new_diamonds,
            buttons_pressed=new_buttons_pressed,
            doors_open=new_doors_open,
            keys=state.keys,
            cost=state.cost + 2,
            spikes_down=state.spikes_down,
            has_key=state.has_key,
            holes_closed=new_holes_closed
        )

    def _update_doors(self, buttons_pressed, rocks):
        """Update door states based on button presses"""
        doors_open = set()

        # Check all pressed positions
        all_pressed = set()
        for button in self.all_buttons:
            if button in buttons_pressed or button in rocks:
                all_pressed.add(button)

        # Open corresponding doors
        for button in all_pressed:
            if button in self.button_door_map:
                doors_open.update(self.button_door_map[button])

        return frozenset(doors_open)

    def heuristic(self, state):
        """A* heuristic function"""
        if self.is_goal(state):
            return 0

        total = 0

        # Distance to uncollected diamonds
        for diamond in state.diamonds:
            total += abs(state.player_pos.x - diamond.x) + abs(state.player_pos.y - diamond.y)

        # Distance to exit (if all diamonds collected)
        if not state.diamonds:
            total += abs(state.player_pos.x - self.exit_pos.x) + abs(state.player_pos.y - self.exit_pos.y)
        else:
            # Encourage diamond collection
            total += len(state.diamonds) * 5

        return total

    def is_goal(self, state):
        """Check if state achieves the goal"""
        return not state.diamonds and state.player_pos == self.exit_pos

    def solve(self):
        """Solve the puzzle using A* algorithm"""
        print("Starting A* search...")

        # Priority queue: (f_score, counter, state)
        open_set = []
        self.counter = 0

        initial_f = self.heuristic(self.initial_state)
        heapq.heappush(open_set, (initial_f, self.counter, self.initial_state))
        self.counter += 1

        closed_set = set()
        came_from = {}
        g_score = {self.initial_state: 0}

        nodes_explored = 0

        while open_set:
            current_f, _, current_state = heapq.heappop(open_set)
            nodes_explored += 1

            if nodes_explored % 100000 == 0:
                print(f"Explored {nodes_explored} nodes, f-score: {current_f}")

            if current_state in closed_set:
                continue

            closed_set.add(current_state)

            if self.is_goal(current_state):
                print(f"Solution found! Explored {nodes_explored} nodes")
                level_id = f"{self.rows}x{self.cols}_exit_{self.exit_pos}"
                save_button_door_map(level_id, self.button_door_map)
                return self._reconstruct_path(came_from, current_state)

            # Generate and explore neighbors
            for neighbor_state, move_cost in self.get_neighbors(current_state):
                if neighbor_state in closed_set:
                    continue

                tentative_g = g_score[current_state] + move_cost

                if neighbor_state not in g_score or tentative_g < g_score[neighbor_state]:
                    came_from[neighbor_state] = current_state
                    g_score[neighbor_state] = tentative_g
                    f_score = tentative_g + self.heuristic(neighbor_state)
                    heapq.heappush(open_set, (f_score, self.counter, neighbor_state))
                    self.counter += 1

        print("No solution found!")
        return None

    def _reconstruct_path(self, came_from, current_state):
        """Reconstruct the solution path"""
        path = [current_state]
        while current_state in came_from:
            current_state = came_from[current_state]
            path.append(current_state)
        path.reverse()
        return path


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
        self.move_delay = 0.16  # Delay between moves (adjust based on game speed)
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

            # Wait longer for door animations
            if len(next_state.doors_open) > len(current_state.doors_open):
                print("  (Door opened - waiting)")
                time.sleep(0.5)

        print("Action sequence executed successfully")
        return True


# Screen Capture
class ScreenCapture:
    """Handles screen capture for game state analysis"""

    def __init__(self):
        self.game_region = self.get_game_region()
        print("Screen capture module initialized")

    @staticmethod
    def get_game_region():
        from utils import find_all_templates
        from initial_config import TEMPLATE_PATH

        screen = pyautogui.screenshot()
        screen = np.array(screen)
        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
        # 2. Detección del template (debe aparecer una sola vez)

        matches = find_all_templates(screen, TEMPLATE_PATH, threshold=0.9)
        if not matches:
            raise Exception("Template no encontrada")

        # Tomar solo el primer resultado
        x1, y1, w, h = matches[0]


        # 3. Define el área de la cuadrícula
        rect_start = (x1 + GRID_OFFSET, y1 + GRID_OFFSET)
        # save_grid_cells(screen, rect_start, RECT_SIZE, GRID_DIMENSIONS)

        a = np.array(rect_start)
        b = np.array(RECT_SIZE)

        return int(a[0]), int(a[1]), int(b[0]), int(b[1])

    def capture_game_area(self):
        """Capture the game screen"""
        print("Capturing game screen...")
        if self.game_region:
            screenshot = pyautogui.screenshot(region=self.game_region)
        else:
            screenshot = pyautogui.screenshot()

        screenshot = np.array(screenshot)
        return cv2.cvtColor(screenshot, cv2.COLOR_BGR2RGB)
# Utility function to print game state
def print_game_state(state):

    """Visualize the game state in the console"""

    grid = state['grid']
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0

    # Map game objects to display characters
    display_map = {
        LevelObjects.WALL: CellDisplay.WALL,
        LevelObjects.PLAYER_LEFT: CellDisplay.PLAYER,
        LevelObjects.PLAYER_RIGHT: CellDisplay.PLAYER,
        LevelObjects.ROCK: CellDisplay.ROCK,
        LevelObjects.BUTTON: CellDisplay.BUTTON,
        LevelObjects.DOOR_CLOSED: CellDisplay.DOOR_CLOSED,
        LevelObjects.DOOR_OPEN: CellDisplay.DOOR_OPEN,
        LevelObjects.DIAMOND: CellDisplay.DIAMOND,
        LevelObjects.KEY: CellDisplay.KEY,
        LevelObjects.KEY_DOOR: CellDisplay.KEY_DOOR,
        LevelObjects.LAVA: CellDisplay.LAVA,
        LevelObjects.SPIKES_UP: CellDisplay.SPIKES_UP,
        LevelObjects.SPIKES_DOWN: CellDisplay.SPIKES_DOWN,
        LevelObjects.EMPTY: CellDisplay.EMPTY,
        LevelObjects.EXIT_CLOSED: CellDisplay.EXIT_CLOSED,
        LevelObjects.EXIT_OPEN: CellDisplay.EXIT_OPEN,
        LevelObjects.HOLE: CellDisplay.HOLE,
    }

    # Top border
    print('  +' + '-' * (2 * cols) + '+')

    for i, row in enumerate(grid):
        row_str = ' '.join([display_map[cell] for cell in row])
        print(f"{str(i).rjust(2)}|{row_str}|")

    # Bottom border
    print('  +' + '-' * (2 * cols) + '+')

    # Column labels
    col_labels = '   ' + ' '.join([str(i % 10) for i in range(cols)])
    print(col_labels)


# Convert vision output to solver input
def convert_to_solver_state(game_state):
    """Convert vision output to solver input"""

    spikes_down = []

    for spikes in game_state['spikes']:
        if not game_state['spikes'][spikes]:
            spikes_down.append(spikes)

    initial_state = GameState(
        player_pos=game_state['player_pos'],
        rocks=frozenset(game_state['rocks']),
        diamonds=frozenset(game_state['diamonds']),
        buttons_pressed=frozenset(),
        doors_open=frozenset(),
        spikes_down=frozenset(spikes_down),
        keys=frozenset(game_state['keys']),
        cost=0,
        has_key=False,
        holes_closed=frozenset(),
    )

    return initial_state


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
