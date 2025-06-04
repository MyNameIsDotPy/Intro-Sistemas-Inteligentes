import numpy as np
from position import Position
# --- Configuración de parámetros ---
TEMPLATE_PATH = "./templates/header_template.png"
SCREENSHOT_PATH = "screen_state.png"
GRID_OFFSET = 27
RECT_SIZE = (341, 511)
GRID_DIMENSIONS = (12, 8)
SINGLE_GRID_SIZE = RECT_SIZE[1] / GRID_DIMENSIONS[0], RECT_SIZE[0] / GRID_DIMENSIONS[1]
DELAY_SECONDS = 3
GROUND_COLOR_BGR = np.array((33, 46, 74))
WALL_COLOR_BGR = np.array((89, 125, 185))
# Movement directions
DIRECTIONS = [
    Position(0, -1),  # UP
    Position(1, 0),  # RIGHT
    Position(0, 1),  # DOWN
    Position(-1, 0),  # LEFT
]