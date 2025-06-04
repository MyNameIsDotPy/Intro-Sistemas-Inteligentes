# Screen Capture
import cv2
import numpy as np
import pyautogui
from initial_config import RECT_SIZE, GRID_OFFSET

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
