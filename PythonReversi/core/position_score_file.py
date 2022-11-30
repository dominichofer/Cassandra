from .position_score import PositionScore
from pathlib import Path
from collections.abc import Iterable


def parse_position_score_file(file_path: Path|str) -> list[PositionScore]:
    if not isinstance(file_path, Path):
        file_path = Path(file_path)
    return [PositionScore.from_string(line) for line in file_path.read_text().strip().split('\n')]


def write_position_score_file(pos_score, file_path: Path|str):
    if not isinstance(pos_score, Iterable):
        pos_score = [pos_score]
    if not isinstance(file_path, Path):
        file_path = Path(file_path)
    file_path.write_text('\n'.join(str(ps) for ps in pos_score))

