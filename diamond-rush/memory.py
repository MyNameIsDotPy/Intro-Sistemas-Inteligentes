import json
import os
from pathlib import Path
from typing import Dict, Set
from common import Position  # Asegúrate de importar correctamente

MEMORY_FOLDER = Path("memory_data")
MEMORY_FOLDER.mkdir(exist_ok=True)

def _pos_to_str(pos: Position) -> str:
    return f"({pos.x},{pos.y})"

def _str_to_pos(s: str) -> Position:
    x, y = map(int, s.strip("()").split(","))
    return Position(x, y)

def save_button_door_map(level_id: str, mapping: Dict[Position, Set[Position]]):
    data = {
        "level_id": level_id,
        "button_door_map": {
            _pos_to_str(btn): [_pos_to_str(door) for door in doors]
            for btn, doors in mapping.items()
        }
    }
    with open(MEMORY_FOLDER / f"{level_id}.json", "w") as f:
        json.dump(data, f, indent=2)

def load_button_door_map(level_id: str) -> Dict[Position, Set[Position]]:
    path = MEMORY_FOLDER / f"{level_id}.json"
    if not path.exists():
        return {}
    with open(path) as f:
        data = json.load(f)
    return {
        _str_to_pos(btn): set(_str_to_pos(door) for door in doors)
        for btn, doors in data["button_door_map"].items()
    }
