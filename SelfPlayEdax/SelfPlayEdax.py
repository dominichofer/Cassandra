import subprocess
import os
import timeit
import unittest
import EdaxOutput
import EdaxScript
from Position import Position, Play, PlayPass, PossibleMoves

class Edax:
    def __init__(self, exe_path):
        self.exe_path = exe_path

    def Solve(self, input_file, level:int):
        output = subprocess.run(
            [self.exe_path, '-l', str(level), '-solve', input_file],
            cwd=os.path.dirname(self.exe_path),
            capture_output=True,
            text=True)
        return EdaxOutput.Parse(output.stdout)

    
def NextPositions(positions, results):
    ret = []
    for pos, res in zip(positions, results):
        if res.pv is None:
            continue #ignore position
        move = EdaxScript.ParseMove(res.pv[0])
        if move is None:
            continue #ignore position
        pos = Play(pos, *move)
        if not PossibleMoves(pos):
            pos = PlayPass(pos)
        ret.append(pos)
    return ret


def SelfPlay(input_file, output_file, level):
    edax = Edax(r'C:\Users\Dominic\Desktop\edax\4.3.2\bin\edax-4.4.exe')
    
    results = edax.Solve(input_file, level)

    file = EdaxScript.File(input_file)
    next = NextPositions([l.pos for l in file.lines], results)

    EdaxScript.WriteToFile(next, output_file)


if __name__ == '__main__':
    #last_file = r'G:\Reversi\edax\e50.script'
    #next_file = r'G:\Reversi\edax\e49_L10.script'
    #SelfPlay(last_file, next_file, 10)


    edax = Edax(r'C:\Users\Dominic\Desktop\edax\4.3.2\bin\edax-4.4.exe')
    for e in range(50):
        for level in [10,15,20]:
            file = EdaxScript.File(fr'G:\Reversi\edax\e{e}_L{level}.script')
            start = timeit.default_timer()
            results = edax.Solve(file.path, 60)
            stop = timeit.default_timer()
            for f, r in zip(file.lines, results):
                f.score = r.score
            file.WriteBack()
            print(f"Solved e{e} L{level} time: {stop - start}")


    #for level in [10,15,20,25,30]:
    #    last_file = r'G:\Reversi\edax\e50.script'
    #    for e in range(49, -1, -1):
    #        next_file = fr'G:\Reversi\edax\e{e}_L{level}.script'
    #        SelfPlay(last_file, next_file, level)
    #        last_file = next_file
