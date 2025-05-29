import time
import cv2
from initial_config import *
import pyautogui as pgui
import numpy as np
from objects.level_objects import LevelObjects

def capture_screen(delay=3, save_path=SCREENSHOT_PATH):
    """Captura la pantalla después de un retardo y la convierte a formato OpenCV."""
    time.sleep(delay)
    screenshot = pgui.screenshot(save_path)
    return cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)


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

def draw_rectangle(image, top_left, size, color=(0, 255, 0), thickness=1):
    """Dibuja un rectángulo en la imagen."""
    bottom_right = (top_left[0] + size[0], top_left[1] + size[1])
    cv2.rectangle(image, top_left, bottom_right, color, thickness)

def draw_grid(image, rect_start, rect_size, grid_size, color=(0, 255, 0), thickness=1):
    """Dibuja una cuadrícula dentro de un rectángulo dado."""
    x0, y0 = rect_start
    width, height = rect_size
    rows, cols = grid_size

    x_step = width / rows
    y_step = height / cols

    # Líneas verticales
    for i in range(1, rows):
        x = int(x0 + x_step * i)
        cv2.line(image, (x, y0), (x, y0 + height), color, thickness)
    # Líneas horizontales
    for j in range(1, cols):
        y = int(y0 + y_step * j)
        cv2.line(image, (x0, y), (x0 + width, y), color, thickness)

def print_game_state(game):
    # Transponer la matriz
    transposed = [list(row) for row in zip(*game)]
    rows = len(transposed)
    cols = len(transposed[0]) if rows > 0 else 0

    # Top border
    print('  +' + '-' * (2 * cols) + '+')

    for i, row in enumerate(transposed):
        row_str = ' '.join(row)
        print(f"{str(i).rjust(2)}|{row_str}|")

    # Bottom border
    print('  +' + '-' * (2 * cols) + '+')

    # Column labels
    col_labels = '   ' + ' '.join([str(i % 10) for i in range(cols)])
    print(col_labels)


def get_game_state(game_screen, level_objects = LevelObjects):

    game = [['##' for _ in range(GRID_DIMENSIONS[1])] for _ in range(GRID_DIMENSIONS[0])]
    new_game_screen = game_screen.copy()

    for x in range(GRID_DIMENSIONS[0]):
        for y in range(GRID_DIMENSIONS[1]):
            a_x, a_y = int(x*SINGLE_GRID_SIZE[1]), int(y*SINGLE_GRID_SIZE[0])
            b_x, b_y = a_x + int(SINGLE_GRID_SIZE[1]), a_y + int(SINGLE_GRID_SIZE[0])

            # cv2.rectangle(new_game_screen, (a_x, a_y), (b_x, b_y), (255, 0, 255), 2)

            crop = new_game_screen[a_y:b_y, a_x:b_x]
            color_mean_bgr = cv2.mean(crop)[:3]
            color = np.ones((100,100,3)) * color_mean_bgr/255


            diff = np.linalg.norm(color_mean_bgr - GROUND_COLOR_BGR)

            if  diff > 50:
                game[x][y] = "WA"
                print("WALL")


    for levelObject in level_objects:
        name = ''.join(list(map(lambda x: x[0], levelObject.name.split('_')))).upper().zfill(2)
        positions = find_all_templates(game_screen, f'./templates/{levelObject.value}')
        for position in positions:
            x = int(position[0]/RECT_SIZE[0] * GRID_DIMENSIONS[0])
            y = int(position[1]/RECT_SIZE[1] * GRID_DIMENSIONS[1])
            game[x][y] = name
            cv2.rectangle(new_game_screen, (position[0], position[1]), (position[0] + position[2], position[1] + position[3]),(0,0,255), 2)
    print_game_state(game)
    return new_game_screen