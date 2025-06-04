import time
import cv2
from initial_config import *
import pyautogui as pgui
import numpy as np
from level_objects import LevelObjects, CellDisplay
from game_state import GameState

def find_all_templates(screen, template_path, threshold=0.8, method=cv2.TM_CCOEFF_NORMED):
    """Encuentra todas las coincidencias del template que superen el threshold"""
    template = cv2.imread(template_path)
    if template is None:
        raise FileNotFoundError(f"Template no encontrado en {template_path}")

    h, w = template.shape[:2]
    result = cv2.matchTemplate(screen, template, method)

    # Obtener todas las posiciones que superen el umbral
    loc = np.where(result >= threshold)

    # Convertir a lista de coordenadas (x,y) de la esquina superior izquierda
    points = list(zip(*loc[::-1]))

    # Crear bounding boxes y aplicar Non-Maximum Suppression
    rects = [(x, y, w, h) for (x, y) in points]
    boxes = non_max_suppression(np.array(rects))

    return boxes


def non_max_suppression(boxes, overlapThresh=0.4):
    """Implementa supresión de no máximos para eliminar detecciones solapadas"""
    if len(boxes) == 0:
        return []

    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    pick = []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = x1 + boxes[:, 2]
    y2 = y1 + boxes[:, 3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:last]]

        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlapThresh)[0])))

    return boxes[pick].astype("int").tolist()

def detectar_piedras_redondas(img, mostrar_resultado=True):
    # Cargar la imagen
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Aplicar desenfoque para reducir ruido
    gris_suavizado = cv2.medianBlur(gris, 5)

    # Detección de círculos con Hough
    circulos = cv2.HoughCircles(gris_suavizado,
                                cv2.HOUGH_GRADIENT,
                                dp=1.2,
                                minDist=30,
                                param1=50,
                                param2=30,
                                minRadius=10,
                                maxRadius=17)

    coordenadas = []
    if circulos is not None:
        circulos = np.uint16(np.around(circulos))
        for (x, y, r) in circulos[0, :]:
            coordenadas.append((x, y, r))
            if mostrar_resultado:
                cv2.circle(img, (x, y), r, (0, 255, 0), 2)
                cv2.circle(img, (x, y), 2, (0, 0, 255), 3)

    if mostrar_resultado:
        cv2.imshow('Piedras Redondas Detectadas', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return coordenadas

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