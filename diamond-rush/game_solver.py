import heapq
from game_state import GameState
from level_objects import LevelObjects
from initial_config import DIRECTIONS

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
        """Discover button-door relationships dynamically by finding closest button-door pairs"""
        mapping = {}
        used_buttons = set()
        used_doors = set()

        # Create list of all possible door-button pairs with distances
        pairs = []
        for door in self.all_doors:
            for button in self.all_buttons:
                # Calculate Manhattan distance
                distance = abs(door.x - button.x) + abs(door.y - button.y)
                pairs.append((distance, door, button))

        # Sort pairs by distance (closest first), then by coordinates
        pairs.sort(key=lambda x: (x[0], x[1].x, x[1].y, x[2].x, x[2].y))

        # Assign closest available pairs
        for _, door, button in pairs:
            if door not in used_doors and button not in used_buttons:
                mapping[button] = {door}
                used_doors.add(door)
                used_buttons.add(button)

        return mapping

    def is_valid_position(self, pos):
        """Check if position is within grid bounds"""
        return 0 <= pos.x < self.cols and 0 <= pos.y < self.rows

    def is_passable(self, pos, state: GameState, is_rock: bool = False):
        """Check if position can be moved to"""
        if not self.is_valid_position(pos):
            return False

        cell = self.grid[pos.y][pos.x]

        # Impassable objects
        if cell is LevelObjects.WALL:
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
        if pos in state.rocks:
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
        new_buttons_pressed = frozenset()
        for button in self.all_buttons:
            if button in state.rocks:
                new_buttons_pressed = new_buttons_pressed | {button}

        self._update_doors(new_buttons_pressed, state.rocks)

        # Update doors
        new_doors_open = state.doors_open
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
        new_rocks = state.rocks

        # Update rock positions
        new_rocks = new_rocks - {rock_old_pos} | {rock_new_pos}

        if rock_cell is LevelObjects.HOLE and rock_new_pos not in new_holes_closed:
            new_holes_closed = new_holes_closed | {rock_new_pos}
            new_rocks = new_rocks - {rock_new_pos}

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

    @staticmethod
    def _reconstruct_path(came_from, current_state):
        """Reconstruct the solution path"""
        path = [current_state]
        while current_state in came_from:
            current_state = came_from[current_state]
            path.append(current_state)
        path.reverse()
        return path
