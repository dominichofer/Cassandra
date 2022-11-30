from .game import Game
from pathlib import Path
from collections.abc import Iterable


def parse_game_file(file_path: Path|str) -> list[Game]:
    if not isinstance(file_path, Path):
        file_path = Path(file_path)
    return [Game.from_string(line) for line in file_path.read_text().strip().split('\n')]


def write_game_file(games, file_path: Path|str):
    if not isinstance(games, Iterable):
        games = [games]
    if not isinstance(file_path, Path):
        file_path = Path(file_path)
    file_path.write_text('\n'.join(str(g) for g in games))
