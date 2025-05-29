from typing import List, Tuple
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder
from pathfinding.core.diagonal_movement import DiagonalMovement

def rastrear_camino(objetivos: List, matrix: List, inicial: Tuple, final: Tuple):
    grid = Grid(matrix=matrix)
    y_inicial, x_inicial = inicial
    start = grid.node(y_inicial, x_inicial)
    ruta_total = []

    for i in range(len(objetivos)):  # Para cada objetivo (diamante)
        y, x = objetivos[i]
        end = grid.node(y, x)

        if i != 0:
            y_pasado, x_pasado = objetivos[i - 1]
            start = grid.node(y_pasado, x_pasado)

        finder = AStarFinder(diagonal_movement=DiagonalMovement.never)
        path, runs = finder.find_path(start, end, grid)
        ruta_total += path

        print('operations:', runs, 'path length:', len(path))
        print(grid.grid_str(path=path, start=start, end=end))

    # Último tramo: del último objetivo hasta el nodo final
    y_ultimo, x_ultimo = objetivos[-1]
    start = grid.node(y_ultimo, x_ultimo)
    y_final, x_final = final
    end = grid.node(y_final, x_final)

    finder = AStarFinder(diagonal_movement=DiagonalMovement.never)
    path, runs = finder.find_path(start, end, grid)
    ruta_total += path

    print('operations:', runs, 'path length:', len(path))
    print(grid.grid_str(path=path, start=start, end=end))

    return ruta_total



matrix=[
	[0,0,0,0,0,0,0,0,0,0],
	[0,1,1,0,0,0,0,0,0,0],
	[0,1,1,0,0,0,0,0,0,0],
	[0,1,1,1,1,1,1,1,1,0],
	[0,0,0,0,0,0,0,1,1,0],
	[0,0,0,0,0,0,0,1,1,0],
	[0,1,1,1,1,1,1,1,1,0],
	[0,1,1,1,0,0,0,0,0,0],
	[0,1,1,1,0,0,0,0,0,0],
	[0,0,0,1,1,1,1,1,1,0],
	[0,0,0,1,1,0,1,1,1,0],
	[0,1,1,1,1,0,1,1,1,0],
	[0,0,0,0,0,0,0,0,0,0]
]

objetivos=[(3,3),(4,3),(5,3),(7,4),(7,5),(5,6),(4,6),(2,6),(2,7),(2,8),(4,9),(5,9)]
inicial=(2,3)
final=(7,10)
grid = Grid(matrix=matrix)
print(grid.grid_str(path=rastrear_camino(objetivos,matrix,inicial,final), start=inicial, end=final))
#print(rastrear_camino(objetivos,matrix,inicial,final))
#primer elemento de la tupla es columna y segundo fila
# end = grid.node(7,11 )

# finder = AStarFinder(diagonal_movement=DiagonalMovement.never)
# path, runs = finder.find_path(start, end, grid)
# print(type(path))
# print('operations:', runs, 'path length:', len(path))
# print(grid.grid_str(path=path, start=start, end=end))


# from pathfinding.core.grid import Grid
# from pathfinding.core.world import World
# from pathfinding.finder.a_star import AStarFinder


# PATH = [
#     # start in the top right on the lower level
#     (2, 0, 0),
#     # then move left
#     (1, 0, 0),
#     (0, 0, 0),
#     # and down
#     (0, 1, 0),
#     (0, 2, 0),
#     # and to the right
#     (1, 2, 0),
#     (2, 2, 0),
#     # now we reached the elevator, move to other map
#     (2, 2, 1),
#     # and continue upwards, around the obstacles
#     (2, 1, 1),
#     (2, 0, 1),
#     # now to the left until we reach our goal
#     (1, 0, 1),
#     (0, 0, 1)
# ]


# def test_connect():
#     level0 = [
#         [1, 1, 1],
#         [1, 0, 0],
#         [1, 1, 1]
#     ]
#     level1 = [
#         [1, 1, 1],
#         [0, 0, 1],
#         [1, 1, 1]
#     ]
#     # create Grid instances for both level
#     grid0 = Grid(matrix=level0, grid_id=0)
#     grid1 = Grid(matrix=level1, grid_id=1)

#     grid0.node(2, 2).connect(grid1.node(2, 2))
#     grid1.node(2, 2).connect(grid0.node(2, 2))

#     # create world with both grids
#     world = World({
#         0: grid0,
#         1: grid1
#     })

#     finder = AStarFinder()
#     path, _ = finder.find_path(grid0.node(2, 0), grid1.node(0, 0), world)
#     assert [tuple(p) for p in path] == PATH
#     print('operations:', _, 'path length:', len(path))
#     print(world.grid_str(path=path, start=grid0.node(2, 0), end=grid1.node(0, 0)))
#     print(grid1.grid_str(path=path, start=grid1.node(2, 2), end=grid1.node(0, 0)))
# test_connect()
