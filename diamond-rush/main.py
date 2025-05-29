import cv2
from utils import *
from initial_config import *
import os


def main():
    # 1. Captura de pantalla
    screen = capture_screen(DELAY_SECONDS)

    # 2. Detección del template (debe aparecer una sola vez)
    matches = find_all_templates(screen, TEMPLATE_PATH, threshold=0.9)
    if not matches:
        raise Exception("Template no encontrada")

    # Tomar solo el primer resultado
    x1, y1, w, h = matches[0]

    # Cargar el template para mostrarlo después
    template = cv2.imread(TEMPLATE_PATH)

    # 3. Define el área de la cuadrícula
    rect_start = (x1 + GRID_OFFSET, y1 + GRID_OFFSET)
    # save_grid_cells(screen, rect_start, RECT_SIZE, GRID_DIMENSIONS)

    a = np.array(rect_start)
    b = a + np.array(RECT_SIZE)

    # 4. Recorta la pantalla de juego
    game_screen = screen[a[1]:b[1], a[0]:b[0]]
    game_screen = get_game_state(game_screen)
    # 5. Dibuja la cuadrícula y el rectángulo de la zona
    # draw_grid(screen, rect_start, RECT_SIZE, GRID_DIMENSIONS)

    # 6. Muestra las imágenes
    cv2.imshow("Screen with Grid", game_screen)
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
