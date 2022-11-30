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
    def __init__(self, exe_path, level: int):
        self.level: int = level
        self.exe: Path = Path(exe_path)
        self.tmp_file: Path = Path(self.exe.parent / str(secrets.token_hex(16)))

    def name(self, separator: str = ' '):
        return separator.join(['Edax4.4', 'level', str(self.level)])

    def __solve(self, pos) -> Line:
        tmp_file = Path(self.exe.parent / f'temp_{secrets.token_hex(16)}.script')
        write_position_file(pos, tmp_file) # create temp file
        cmd = [self.exe, '-n', '1', '-l', str(self.level), '-solve', tmp_file]
        if self.level < 2:
            cmd += ['-h', '10']
        result = subprocess.run(
            cmd,
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

    #def evaluation_accuracy(self, empty_count: int, d: int):
    #    EVAL_A = -0.10026799
    #    EVAL_B = 0.31027733
    #    EVAL_C = -0.57772603
    #    EVAL_a = 0.07585621
    #    EVAL_b = 1.16492647
    #    EVAL_c = 5.4171698
    #    sigma = EVAL_A * empty_count + EVAL_B * empty_count + EVAL_C * d
    #    sigma = EVAL_a * sigma * sigma + EVAL_b * sigma + EVAL_c
    #    return sigma
