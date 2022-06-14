from typing import List
from pathlib import Path
import subprocess
from Position import Position, PositionScore
import EdaxScript
import EdaxOutput


class Edax:
    def __init__(self, exe: Path, tmp_file = None):
        self.exe: Path = exe
        self.tmp_file: Path = Path(tmp_file or exe.parent / 'tmp.script')

    def Solve(self, input, level: int) -> List[EdaxOutput.Line]:
        if isinstance(input, Path):
            file = input
        else:
            file = self.tmp_file
            EdaxScript.WriteToFile(input, self.tmp_file)

        output = subprocess.run(
            [self.exe, '-l', str(level), '-solve', file],
            cwd = self.exe.parent,
            capture_output = True,
            text = True)
        return EdaxOutput.Parse(output.stdout)
