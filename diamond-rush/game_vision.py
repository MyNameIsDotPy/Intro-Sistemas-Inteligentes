import numpy as np
from utils import detectar_piedras_redondas
from level_objects import LevelObjects
import cv2
from initial_config import GROUND_COLOR_BGR, SINGLE_GRID_SIZE, GRID_DIMENSIONS
from position import Position

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

                if diff > 60:
                    grid[y][x] = LevelObjects.WALL

        # Detect all rocks
        piedras = detectar_piedras_redondas(screenshot, mostrar_resultado=False)

        for piedra in piedras:
            x, y, r = piedra
            x = x//cell_size
            y = y//cell_size
            print(x, y, r)
            if x >= cols or y >= rows or r <= 12:
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
