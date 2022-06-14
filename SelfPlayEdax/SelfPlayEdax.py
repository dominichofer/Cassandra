import subprocess
import os
import timeit
import random
from pathlib import Path
import EdaxOutput
import EdaxScript
from Edax import Edax
from Position import *
from Game import Game, WriteToFile
from Database import DataBase, ParsePosition

#"""
#SELECT ll.name, l.depth, l.confidence, rr.name, r.depth, r.confidence, avg((l.score - r.score)*(l.score - r.score)) - avg(l.score - r.score)*avg(l.score - r.score) FROM Evaluation AS l
#INNER JOIN Evaluation AS r ON l.pos_id = r.pos_id
#INNER JOIN Position ON Position.id = l.pos_id AND l.depth = empty_count
#INNER JOIN Evaluator AS ll ON ll.id = l.evaluator_id
#INNER JOIN Evaluator AS rr ON rr.id = r.evaluator_id
#GROUP BY l.evaluator_id, l.depth, l.confidence, r.evaluator_id, r.depth, r.confidence
#"""

if __name__ == '__main__':
    edax = Edax(Path(r'C:\Users\Dominic\Desktop\edax\4.3.2\bin\edax-4.4.exe'))
    db = DataBase(r'G:\Reversi\DB.db')
    #db.CreateTables()
    #db.AddEvaluator('RND')
    #db.AddEvaluator('Edax4.4')
    #db.AddEvaluator('Cassandra')

    #for d in range(11):
    #    start = timeit.default_timer()
    #    db.CreateGroupWithPositions(f'all_unique_e{60-d}', AllUniqueChildren(Position.Start(), d))
    #    stop = timeit.default_timer()
    #    print(stop - start)

    #all_unique_e50 = db.GetGroup('all_unique_e50')

    #for level in [10, 15, 20]:
    #    games = [Game(pos) for pos in random.sample(all_unique_e50, 1_000)]

    #    for e in range(50):
    #        start = timeit.default_timer()
    #        results = edax.Solve([game.Position() for game in games], level)
    #        for game, result in zip(games, results):
    #            if result.pv and result.pv[0] != 64:
    #                game.Play(result.pv[0])
    #        stop = timeit.default_timer()
    #        print(f'e{e} {stop - start}')

    #    for game in games:
    #        db.AddGame(game, 'Edax4.4', str(level), 'Edax4.4', str(level))
    
    #db.Commit()
    
    games = db.GetGames('Edax4.4', '10', 'Edax4.4', '10') + db.GetGames('Edax4.4', '15', 'Edax4.4', '15') + db.GetGames('Edax4.4', '20', 'Edax4.4', '20')
    for depth in [3,4,5]:
        for e in [26,27,28]:
            pos = [deepcopy(p) for game in games for p in game.Positions() if p.EmptyCount() == e]
            start = timeit.default_timer()
            results = edax.Solve(pos, depth)
            for p, r in zip(pos, results):
                db.AddPosition(p)
                db.AddEvaluationToPosition(p, 'Edax4.4', r.depth, r.confidence, r.score)
            stop = timeit.default_timer()
            db.Commit()
            print(f'EmptyCount{e} {stop - start}')

        #EdaxScript.WriteToFile(
        #    set(child.FlipToUnique() for child in Children(Position.Start(), d)),
        #    Path(fr'C:\Users\Dominic\Desktop\edax\4.3.2\bin\problem\e{60-d}_all.script'))
    
    #last_file = r'G:\Reversi\edax\e50.script'
    #next_file = r'G:\Reversi\edax\e49_L10.script'
    #SelfPlay(last_file, next_file, 10)


    #edax = Edax(r'C:\Users\Dominic\Desktop\edax\4.3.2\bin\edax-4.4.exe')
    #for e in [29]:
    #    for level in [10]:
    #        file = EdaxScript.File(fr'G:\Reversi\edax\e{e}_L{level}.script')
    #        start = timeit.default_timer()
    #        results = edax.Solve(file.path, 60)
    #        stop = timeit.default_timer()
    #        for f, r in zip(file.lines, results):
    #            f.score = r.score
    #        file.WriteBack()
    #        print(f"Solved e{e} L{level} time: {stop - start}")


    #for level in [10,15,20,25,30]:
    #    last_file = r'G:\Reversi\edax\e50.script'
    #    for e in range(49, -1, -1):
    #        next_file = fr'G:\Reversi\edax\e{e}_L{level}.script'
    #        SelfPlay(last_file, next_file, level)
    #        last_file = next_file
