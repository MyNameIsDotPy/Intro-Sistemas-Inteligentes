from typing import List
import heapq

class Node:
    def __init__(self, estado, padre, movimiento, g, h):
        self.estado: List[List] = estado # Tablero: List[List] [3x3]
        self.padre = padre
        self.movimiento = movimiento # Nuevo estado después de moverse
        self.g = g # Costo real acumulado
        self.h = h # Heurística
        self.f = g + h # Costo total

    def __lt__(self, other):
        return self.f < other.f

    def __str__(self):
        result = ''
        for row in self.movimiento:
            result += '| ' + ' '.join(map(str, row)) + ' |\n'
        return result


def tuple2list(state):
    final = []
    for row in state:
        final.append(list(row))
    return final

def list2tuple(state):
    final = []
    for row in state:
        final.append(tuple(row))
    return tuple(final)



def dumb_heuristic(state, objective):
    suma = 0
    for row1, row2 in zip(state, objective):
        for a, b in zip(row1, row2):
            suma += 1 if a != b else 0
    return suma

def manhattan_heuristic(state, objective):
    distance = 0
    pos_objetivo = {
        objective[i][j]: (i, j) for i in range(3) for j in range(3)
    }
    for i in range(3):
        for j in range(3):
            valor = state[i][j]
            if valor != 0:
                x, y = pos_objetivo[valor]
                distance += abs(i - x) + abs(j - y)

    return distance

def generar_vecinos(nodo, objetivo, heuristic):
    i, j = [(i, j) for i in range(3) for j in range(3) if nodo.estado[i][j] == 0][0]

    movimientos = []
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for dx, dy in dirs:
        if 0 <= i+dx < 3 and 0 <= j+dy < 3:
            nuevo_estado = tuple2list(nodo.estado)
            # Swap
            nuevo_estado[i][j], nuevo_estado[i+dx][j+dy] = nuevo_estado[i+dx][j+dy], nuevo_estado[i][j]

            nuevo_estado = list2tuple(nuevo_estado)

            h = heuristic(nuevo_estado, objetivo)
            movimientos.append(Node(nuevo_estado, nodo, nuevo_estado, nodo.g + 1, h))

    return movimientos

def a_star(start, objective, heuristic=manhattan_heuristic):
    open_list = []
    closed_list = set()
    moves = 0

    start = list2tuple(start)
    objective = list2tuple(objective)

    nodo_inicial = Node(
        start, None, None, 0, heuristic(start, objective)
    )

    heapq.heappush(open_list, nodo_inicial)

    while open_list:
        moves += 1
        if moves > 10000:
            break
        nodo_actual: Node = heapq.heappop(open_list)

        if nodo_actual.estado == objective:
            camino = []

            while nodo_actual.padre:
                camino.append(nodo_actual)
                nodo_actual = nodo_actual.padre
            return camino[::-1]

        closed_list.add(nodo_actual.estado)

        vecinos = generar_vecinos(nodo_actual, objective, heuristic)

        for vecino in vecinos:
            if vecino.estado in closed_list:
                continue

            in_open = False
            for n in open_list:
                if n.estado == vecino.estado and n.f <= vecino.f:
                    in_open = True
                    break

            if not in_open:
                heapq.heappush(open_list, vecino)
    return None

INITIAL_STATE = [
    [5, 1, 6],
    [7, 4, 2],
    [3, 8, 0],
]

OBJECTIVE = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 0],
]

if __name__ == "__main__":
    solution = a_star(INITIAL_STATE, OBJECTIVE, dumb_heuristic)
    print(f'Se tomaron {len(solution)} pasos para llegar a la solución' if solution else 'Sin solución')
    if solution:
        for step in solution:
            step: Node
            print(step)