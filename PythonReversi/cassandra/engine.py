import subprocess
import secrets
import numpy as np
import multiprocessing
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Iterable
from .output import Line, parse
from core import write_position_file


class Engine:
    def __init__(self, exe_path, model_path: str, level: str):
        self.level: str = str(level)
        self.exe: Path = Path(exe_path)
        self.model: Path = Path(model_path)
        self.tmp_file: Path = Path(self.exe.parent / str(secrets.token_hex(16)))
        
    def name(self, separator: str = ' '):
        return separator.join(['Cassandra', 'level', self.level])

    def __solve(self, pos) -> Line:
        tmp_file = Path(self.exe.parent / f'temp_{secrets.token_hex(16)}.script')
        write_position_file(pos, tmp_file) # create temp file
        result = subprocess.run(
            [self.exe, '-d', self.level, '-m', self.model, '-solve', tmp_file],
            cwd = self.exe.parent,
            capture_output = True,
            text = True)
        tmp_file.unlink() # remove temp file
        return parse(result.stdout)
            
    def solve(self, pos) -> list[Line]:
        if not isinstance(pos, Iterable):
            pos = [pos]
        
        pool = ThreadPool()
        results = pool.map(
            self.__solve, 
            np.array_split(pos, multiprocessing.cpu_count() * 4)
            )
        pool.close()
        return [r for result in results for r in result]

    def choose_move(self, pos) -> list[int]:
        result = self.solve(pos)
        return [(r.pv[0] if r.pv else 64) for r in result]