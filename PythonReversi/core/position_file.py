from .position import Position
from pathlib import Path
from collections.abc import Iterable


def parse_position_file(file_path: Path|str) -> list[Position]:
    if not isinstance(file_path, Path):
        file_path = Path(file_path)
    return [Position.from_string(line) for line in file_path.read_text().strip().split('\n')]


def write_position_file(pos, file_path: Path|str):
    if not isinstance(pos, Iterable):
        pos = [pos]
    if not isinstance(file_path, Path):
        file_path = Path(file_path)
    file_path.write_text('\n'.join(str(p) for p in pos))
