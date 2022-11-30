from .game_score import GameScore
from pathlib import Path
from collections.abc import Iterable
from itertools import chain


def parse_game_score_file(file_path: Path|str) -> list[GameScore]:
    if not isinstance(file_path, Path):
        file_path = Path(file_path)

    return [GameScore.from_string(line) for line in file_path.read_text().strip().split('\n')]


def write_game_score_file(game_score, file_path: Path|str):
    if not isinstance(game_score, Iterable):
        game_score = [game_score]
    if not isinstance(file_path, Path):
        file_path = Path(file_path)

    file_path.write_text('\n'.join(str(x) for x in game_score))
