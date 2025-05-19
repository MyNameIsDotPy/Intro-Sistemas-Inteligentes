import time
import cv2
import numpy as np
import pyautogui as pgui
import os

# --- Configuración de parámetros ---
TEMPLATE_PATH = "./templates/header_template.png"
SCREENSHOT_PATH = "screen_state.png"
GRID_OFFSET = 27
RECT_SIZE = (341, 511)
GRID_DIMENSIONS = (8, 12)
DELAY_SECONDS = 3

def capture_screen(delay=3, save_path=SCREENSHOT_PATH):
    """Captura la pantalla después de un retardo y la convierte a formato OpenCV."""
    time.sleep(delay)
    screenshot = pgui.screenshot(save_path)
    return cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

def find_template(screen, template_path):
    """Busca la posición del template en la pantalla capturada."""
    template = cv2.imread(template_path)
    if template is None:
        raise FileNotFoundError(f"No se encontró el template en {template_path}")
    res = cv2.matchTemplate(screen, template, cv2.TM_SQDIFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    return min_loc, template

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

def main():
    # 1. Captura de pantalla
    screen = capture_screen(DELAY_SECONDS)

    # 2. Detección del template
    min_loc, template = find_template(screen, TEMPLATE_PATH)
    x1, y1 = min_loc
    x2, y2 = x1 + template.shape[1], y1 + template.shape[0]

    # 3. Define el área de la cuadrícula
    rect_start = (x1 + GRID_OFFSET, y1 + GRID_OFFSET)
    save_grid_cells(screen, rect_start, RECT_SIZE, GRID_DIMENSIONS)

    # 4. Dibuja el rectángulo del template detectado
    cv2.rectangle(screen, (x1, y1), (x2, y2), (0, 255, 0), 3)

    # 5. Dibuja la cuadrícula y el rectángulo de la zona
    draw_grid(screen, rect_start, RECT_SIZE, GRID_DIMENSIONS)
    draw_rectangle(screen, rect_start, RECT_SIZE)

    # 6. Muestra las imágenes
    cv2.imshow("Screen with Grid", screen)
    cv2.imshow("Template", template)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def save_grid_cells(image, rect_start, rect_size, grid_size, output_dir="./grid_cells"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    x0, y0 = rect_start
    width, height = rect_size
    rows, cols = grid_size

    x_step = width / rows
    y_step = height / cols

    for i in range(rows):
        for j in range(cols):
            x_start = int(x0 + x_step * i)
            y_start = int(y0 + y_step * j)
            x_end = int(x0 + x_step * (i + 1))
            y_end = int(y0 + y_step * (j + 1))

            cell_img = image[y_start:y_end, x_start:x_end]
            filename = os.path.join(output_dir, f"cell_{i+1}_{j+1}.png")
            cv2.imwrite(filename, cell_img)

    print(f"Saved {rows*cols} grid cell images to '{output_dir}'")


if __name__ == "__main__":
    main()
