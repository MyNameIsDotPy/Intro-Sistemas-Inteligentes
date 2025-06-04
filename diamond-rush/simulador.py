from main import GameState, DiamondRushSolver, LevelObjects
from common import Position

def simular_nivel_14():
    # Dimensiones del nivel
    rows, cols = 11, 9
    grid = [[LevelObjects.EMPTY for _ in range(cols)] for _ in range(rows)]

    # Posiciones clave (según imagen)
    player_pos = Position(4, 10)
    exit_pos = Position(4, 0)

    rocks = {
        Position(4, 6), Position(6, 5), Position(3, 7),
        Position(5, 9), Position(3, 3)
    }

    diamonds = {
        Position(4, 3), Position(3, 4), Position(2, 1),
        Position(5, 4), Position(2, 4), Position(4, 4),
        Position(6, 6), Position(6, 8), Position(2, 2)
    }

    buttons = [Position(1, 7)]  # botón verde visible
    doors = {
        Position(3, 1): False, Position(5, 2): False,
        Position(2, 5): False, Position(4, 8): False  # esta es de llave
    }

    keys = [Position(8, 10)]
    key_doors = [Position(4, 8)]

    # Pintar en grid (opcional)
    grid[player_pos.y][player_pos.x] = LevelObjects.PLAYER_RIGHT
    for d in diamonds:
        grid[d.y][d.x] = LevelObjects.DIAMOND
    for r in rocks:
        grid[r.y][r.x] = LevelObjects.ROCK
    for b in buttons:
        grid[b.y][b.x] = LevelObjects.BUTTON
    for k in keys:
        grid[k.y][k.x] = LevelObjects.KEY
    for d in doors:
        grid[d.y][d.x] = LevelObjects.KEY_DOOR if d in key_doors else LevelObjects.DOOR_CLOSED
    grid[exit_pos.y][exit_pos.x] = LevelObjects.EXIT_CLOSED

    # Estructura de visión artificial simulada
    game_state = {
        "grid": grid,
        "player_pos": player_pos,
        "rocks": list(rocks),
        "diamonds": list(diamonds),
        "buttons": buttons,
        "doors": doors,
        "exit": exit_pos,
        "exit_open": False,
        "keys": keys,
        "key_doors": key_doors,
        "spikes": {}
    }

    initial_state = GameState(
        player_pos=player_pos,
        rocks=frozenset(rocks),
        diamonds=frozenset(diamonds),
        buttons_pressed=frozenset(),
        doors_open=frozenset(),
        keys=frozenset(keys),
        cost=0,
        has_key=False,
        spikes_down=frozenset(),
        holes_closed=frozenset()
    )

    # Ejecutar solver
    solver = DiamondRushSolver(initial_state, game_state)
    solver.button_door_map = {
        Position(3, 9): {Position(3, 8)},  # 1 → 1
        Position(1, 7): {Position(1, 6)},  # 2 → 2
        Position(3, 6): {Position(3, 5)},  # 3 → 3
        Position(1, 3): {Position(1, 2)}   # 4 → 4
    }    
    path = solver.solve()

    if path:
        print(f"✔ Solución encontrada con {len(path)} pasos")
        for step in path:
            print(step.player_pos)
    else:
        print("❌ No se encontró solución")

simular_nivel_14()
