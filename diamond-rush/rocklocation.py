import cv2
import numpy as np

def detectar_piedras_redondas(imagen_path, mostrar_resultado=True):
    # Cargar la imagen
    imagen = cv2.imread(imagen_path)
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    # Aplicar desenfoque para reducir ruido
    gris_suavizado = cv2.medianBlur(gris, 5)

    # Detección de círculos con Hough
    circulos = cv2.HoughCircles(gris_suavizado, 
                                cv2.HOUGH_GRADIENT, 
                                dp=1.2, 
                                minDist=30,
                                param1=50, 
                                param2=30, # parametros relevantes-> este es el grado de seguridad que debe tener para afirmar que es una piedra
                                minRadius=8,# parametros relevantes-> radio minimo
                                maxRadius=15) # parametros relevantes-> radio maximo  

    coordenadas = []
    if circulos is not None:
        circulos = np.uint16(np.around(circulos))
        for (x, y, r) in circulos[0, :]:
            coordenadas.append((x, y, r))
            if mostrar_resultado:
                cv2.circle(imagen, (x, y), r, (0, 255, 0), 2)
                cv2.circle(imagen, (x, y), 2, (0, 0, 255), 3)

    if mostrar_resultado:
        cv2.imshow('Piedras Redondas Detectadas', imagen)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return coordenadas

coords = detectar_piedras_redondas("dmrush3.PNG")
print("Coordenadas de piedras redondas:", coords)
