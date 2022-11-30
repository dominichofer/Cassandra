from core import *
import edax
import cassandra
import timeit
import random
import statistics
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from pathlib import Path


#def add_self_play_to_db(db: DataBase, pos: list[Position], engine, level):
#    start = timeit.default_timer()
#    games = self_play(engine, [Game(p) for p in pos], level)
#    db.add_games_and_positions(games, engine.name, str(level), engine.name, str(level))
#    db.commit()
#    stop = timeit.default_timer()
#    print(f'Self played {engine.name} level {level} in {stop - start}.')


#def eval_accuracy(db: DataBase, evaluator: str, player: str, first_level: int, opponent: str, second_level: int):
#    evaluator_id = db.get_evaluator_id(evaluator)
#    first_eval_id = db.get_evaluator_id(player)
#    second_eval_id = db.get_evaluator_id(opponent)
#    db.execute('''
#        SELECT big.depth, count(*), avg(small.score - big.score), avg((small.score - big.score)*(small.score - big.score))
#        FROM Evaluation small, Evaluation big, Game
#        LEFT JOIN Position p ON big.pos_id = p.id AND big.depth = p.empty_count
#        INNER JOIN PositionToGame ptg ON ptg.pos_id = p.id AND ptg.game_id = Game.id
#        WHERE Game.first_eval_id = ? and Game.first_level = ?
#        AND Game.second_eval_id = ? and Game.second_level = ?
#        AND small.pos_id = big.pos_id
#        AND (small.depth < big.depth OR (small.depth = big.depth AND small.confidence < big.confidence))
#        AND small.evaluator_id = ? AND big.evaluator_id = ?
#        AND small.depth = 0 AND small.confidence = 1e10000
#        AND big.confidence = 1e10000
#        GROUP BY big.depth
#        ORDER BY big.depth ASC
#        ''', (first_eval_id, first_level, second_eval_id, second_level, evaluator_id, evaluator_id))
#    data = db.fetch_all()
#    x = [d[0] for d in data]
#    y = [2 * math.sqrt(d[3] - d[2]*d[2]) for d in data]
#    yerr = [0.05 * t for t in y]
#    return x, y, yerr

#def score_histogram(db: DataBase, evaluator: str, player: str, first_level: int, opponent: str, second_level: int, empty_count: int):
#    evaluator_id = db.get_evaluator_id(evaluator)
#    first_eval_id = db.get_evaluator_id(player)
#    second_eval_id = db.get_evaluator_id(opponent)
#    db.execute('''
#        SELECT score, count(*) FROM Evaluation e, Game
#        LEFT JOIN Position p ON e.pos_id = p.id AND e.depth = p.empty_count
#        INNER JOIN PositionToGame ptg ON ptg.pos_id = p.id AND ptg.game_id = Game.id
#        WHERE Game.first_eval_id = ? and Game.first_level = ?
#        AND Game.second_eval_id = ? and Game.second_level = ?
#        AND evaluator_id = ?
#        AND p.empty_count = ?
#        GROUP BY score
#        ''', (first_eval_id, first_level, second_eval_id, second_level, evaluator_id, empty_count))
#    data = db.fetch_all()
#    x = [2 * d[0] for d in data]
#    y = [d[1] for d in data]
#    return x, y


def self_play(engine, pos: list[Position], level: int) -> list[Game]:
    for e in range(60):
        best_moves = engine.best_move([g.current_position for g in games], level)
        for game, best_move in zip(games, best_moves):
            if best_move != 64:
                game.play(best_move)
    return games


if __name__ == '__main__':
    random_engine = RandomEngine()
    edax_level_0 = edax.Engine(r'G:\edax-ms-windows\edax-4.4', level=0)
    edax_level_5 = edax.Engine(r'G:\edax-ms-windows\edax-4.4', level=5)
    edax_level_10 = edax.Engine(r'G:\edax-ms-windows\edax-4.4', level=10)
    edax_level_15 = edax.Engine(r'G:\edax-ms-windows\edax-4.4', level=15)
    edax_level_20 = edax.Engine(r'G:\edax-ms-windows\edax-4.4', level=20)
    edax_level_60 = edax.Engine(r'G:\edax-ms-windows\edax-4.4', level=60)

    logistello_10k = cassandra.Engine(r'C:\Users\Dominic\source\repos\Cassandra\bin\Solver.exe', r'G:\Reversi2\rnd_10000_logistello.model', level=0)
    logistello_20k = cassandra.Engine(r'C:\Users\Dominic\source\repos\Cassandra\bin\Solver.exe', r'G:\Reversi2\rnd_20000_logistello.model', level=0)
    logistello_50k = cassandra.Engine(r'C:\Users\Dominic\source\repos\Cassandra\bin\Solver.exe', r'G:\Reversi2\rnd_50000_logistello.model', level=0)
    logistello_100k = cassandra.Engine(r'C:\Users\Dominic\source\repos\Cassandra\bin\Solver.exe', r'G:\Reversi2\rnd_100000_logistello.model', level=0)
    logistello_200k = cassandra.Engine(r'C:\Users\Dominic\source\repos\Cassandra\bin\Solver.exe', r'G:\Reversi2\rnd_200000_logistello.model', level=0)
    logistello_500k = cassandra.Engine(r'C:\Users\Dominic\source\repos\Cassandra\bin\Solver.exe', r'G:\Reversi2\rnd_500000_logistello.model', level=0)
    logistello_1M = cassandra.Engine(r'C:\Users\Dominic\source\repos\Cassandra\bin\Solver.exe', r'G:\Reversi2\rnd_1000000_logistello.model', level=0)

    edax_10k = cassandra.Engine(r'C:\Users\Dominic\source\repos\Cassandra\bin\Solver.exe', r'G:\Reversi2\rnd_10000_edax.model', level=0)
    edax_20k = cassandra.Engine(r'C:\Users\Dominic\source\repos\Cassandra\bin\Solver.exe', r'G:\Reversi2\rnd_20000_edax.model', level=0)
    edax_50k = cassandra.Engine(r'C:\Users\Dominic\source\repos\Cassandra\bin\Solver.exe', r'G:\Reversi2\rnd_50000_edax.model', level=0)
    edax_100k = cassandra.Engine(r'C:\Users\Dominic\source\repos\Cassandra\bin\Solver.exe', r'G:\Reversi2\rnd_100000_edax.model', level=0)
    edax_200k = cassandra.Engine(r'C:\Users\Dominic\source\repos\Cassandra\bin\Solver.exe', r'G:\Reversi2\rnd_200000_edax.model', level=0)
    edax_500k = cassandra.Engine(r'C:\Users\Dominic\source\repos\Cassandra\bin\Solver.exe', r'G:\Reversi2\rnd_500000_edax.model', level=0)
    edax_1M = cassandra.Engine(r'C:\Users\Dominic\source\repos\Cassandra\bin\Solver.exe', r'G:\Reversi2\rnd_1000000_edax.model', level=0)

    cassandra_10k = cassandra.Engine(r'C:\Users\Dominic\source\repos\Cassandra\bin\Solver.exe', r'G:\Reversi2\rnd_10000_cassandra.model', level=0)
    cassandra_20k = cassandra.Engine(r'C:\Users\Dominic\source\repos\Cassandra\bin\Solver.exe', r'G:\Reversi2\rnd_20000_cassandra.model', level=0)
    cassandra_50k = cassandra.Engine(r'C:\Users\Dominic\source\repos\Cassandra\bin\Solver.exe', r'G:\Reversi2\rnd_50000_cassandra.model', level=0)
    cassandra_100k = cassandra.Engine(r'C:\Users\Dominic\source\repos\Cassandra\bin\Solver.exe', r'G:\Reversi2\rnd_100000_cassandra.model', level=0)
    cassandra_200k = cassandra.Engine(r'C:\Users\Dominic\source\repos\Cassandra\bin\Solver.exe', r'G:\Reversi2\rnd_200000_cassandra.model', level=0)
    cassandra_500k = cassandra.Engine(r'C:\Users\Dominic\source\repos\Cassandra\bin\Solver.exe', r'G:\Reversi2\rnd_500000_cassandra.model', level=0)
    cassandra_1M = cassandra.Engine(r'C:\Users\Dominic\source\repos\Cassandra\bin\Solver.exe', r'G:\Reversi2\rnd_1000000_cassandra.model', level=0)
    
    #pos = parse_position_file(r'G:\Reversi2\Edax4.4_level_20_vs_Edax4.4_level_20_1k.pos')
    #lines = edax_level_60.solve(pos)
    #pos_scores = [PositionScore(pos, line.score) for pos, line in zip(pos, lines)]
    #write_position_score_file(pos_scores, r'G:\Reversi2\Edax4.4_level_20_vs_Edax4.4_level_20_1k_eval_Edax4.4_level_60.pos')

    start_pos = parse_position_file(r'G:\Reversi2\all_unique_e54.script')

    #for level in [None, 0, 5, 10, 15, 20]:
    #    # engine
    #    if level is None:
    #        engine = RandomEngine()
    #    else:
    #        engine = edax_engine

    #    # self play
    #    start = timeit.default_timer()
    #    games = self_play(engine, [Game(p) for p in random.sample(all_unique, 1_000)], level)
    #    stop = timeit.default_timer()
    #    print(f'self play {engine.name} {level=}: {stop - start}')

    #    write_game_file(games, fr'G:\Reversi2\{engine.name}_level_{level}_vs_{engine.name}_level_{level}_1k.game')
        
    #    pos = [p for g in games for p in g.positions()]
    #    write_position_file(pos, fr'G:\Reversi2\{engine.name}_level_{level}_vs_{engine.name}_level_{level}_1k.pos')

    #    pos_filtered = [p for p in pos if 1 <= p.empty_count() <= 25]
        
    #    start = timeit.default_timer()
    #    lines = edax_engine.solve(pos_filtered, 0)
    #    pos_scores = [PositionScore(pos, line.score) for pos, line in zip(pos_filtered, lines)]
    #    write_position_score_file(pos_scores, fr'G:\Reversi2\{engine.name}_level_{level}_vs_{engine.name}_level_{level}_1k_e0-e25_eval_edax_level_0.pos')
    #    stop = timeit.default_timer()
    #    print(f'edax level 0: {stop - start}')

    #    start = timeit.default_timer()
    #    lines = edax_engine.solve(pos_filtered, 60)
    #    pos_scores = [PositionScore(pos, line.score) for pos, line in zip(pos_filtered, lines)]
    #    stop = timeit.default_timer()
    #    print(f'edax level 60: {stop - start}')

    #    write_position_score_file(pos_scores, fr'G:\Reversi2\{engine.name}_level_{level}_vs_{engine.name}_level_{level}_1k_e0-e25_eval_edax_level_60.pos')

    evaluators = [
        (edax_level_0, 'Edax level 0'),
        (logistello_10k, 'Logistello patterns, 10k games'),
        (logistello_20k, 'Logistello patterns, 20k games'),
        (logistello_50k, 'Logistello patterns, 50k games'),
        (logistello_100k, 'Logistello patterns, 100k games'),
        (logistello_200k, 'Logistello patterns, 200k games'),
        (logistello_500k, 'Logistello patterns, 500k games'),
        (logistello_1M, 'Logistello patterns, 1M games'),

        (edax_10k, 'Edax patterns, 10k games'),
        (edax_20k, 'Edax patterns, 20k games'),
        (edax_50k, 'Edax patterns, 50k games'), 
        (edax_100k, 'Edax patterns, 100k games'),
        (edax_200k, 'Edax patterns, 200k games'),
        (edax_500k, 'Edax patterns, 500k games'),
        (edax_1M, 'Edax patterns, 1M games'),

        (cassandra_10k, 'Cassandra patterns, 10k games'),
        (cassandra_20k, 'Cassandra patterns, 20k games'),
        (cassandra_50k, 'Cassandra patterns, 50k games'),
        (cassandra_100k, 'Cassandra patterns, 100k games'),
        (cassandra_200k, 'Cassandra patterns, 200k games'),
        (cassandra_500k, 'Cassandra patterns, 500k games'),
        (cassandra_1M, 'Cassandra patterns, 1M games'),
        ]

    self_players = [
        random_engine,
        edax_level_0,
        edax_level_5,
        edax_level_10,
        edax_level_15,
        edax_level_20,
        ]
    
    for evalor, evalor_name in evaluators:
        for player in self_players:
            name = player.name('_')
            ps_exact = parse_position_score_file(fr'G:\Reversi2\{name}_vs_{name}_1k_e0-e25_eval_edax_level_60.pos')
            lines = evalor.solve([ps.pos for ps in ps_exact])
            ps_eval = [PositionScore(ps.pos, line.score) for ps, line in zip(ps_exact, lines)]
            x = []
            y = []
            yerr = []
            for e in range(1, 26):
                diffs = [exact.score - evl.score for exact, evl in zip(ps_exact, ps_eval) if exact.pos.empty_count() == e]
                sd = statistics.stdev(diffs)
                x.append(e)
                y.append(sd)
                yerr.append(0.05 * sd)
            plt.errorbar(x, y, yerr, fmt='o', label=str(player.name()))
        plt.title(evalor_name)
        plt.xlabel('empty fields')
        plt.ylabel('standard deviation')
        plt.xlim(0, 25)
        plt.ylim(0, 12)
        plt.grid(True, which='major', axis='both')
        plt.legend(loc=4)
        plt.savefig(r'G:\Reversi2\{}.png'.format(evalor_name.replace(' ', '_').replace(',', '')))
        plt.clf()
        
    #for level in [None, 0, 5, 10, 15, 20]:
    #    # engine
    #    if level is None:
    #        engine = RandomEngine()
    #    else:
    #        engine = edax_engine
            
    #    pos_scores_level_0 = parse_position_score_file(fr'G:\Reversi2\{engine.name}_level_{level}_vs_{engine.name}_level_{level}_1k_e0-e25_eval_edax_level_0.pos')
    #    pos_scores_level_60 = parse_position_score_file(fr'G:\Reversi2\{engine.name}_level_{level}_vs_{engine.name}_level_{level}_1k_e0-e25_eval_edax_level_60.pos')
    #    x = []
    #    y = []
    #    yerr = []
    #    for e in range(1, 26):
    #        sd = statistics.stdev(ps_60.score - ps_0.score for ps_60, ps_0 in zip(pos_scores_level_60, pos_scores_level_0) if ps_60.pos.empty_count() == e)
    #        x.append(e)
    #        y.append(sd)
    #        yerr.append(0.05 * sd)
    #    plt.errorbar(x, y, yerr, fmt='o', label=f'{engine.fully_qualified_name(level)}')

    #plt.ylim(0, 12)
    #plt.xlabel('empty fields')
    #plt.ylabel('standard deviation')
    #plt.legend(loc=4)
    #plt.show()

    # histogram
    #for level in [None, 0, 5, 10, 15, 20]:
    #    # engine
    #    if level is None:
    #        engine = RandomEngine()
    #    else:
    #        engine = edax_engine

    #    pos_score = parse_position_score_file(fr'G:\Reversi2\{engine.name}_level_{level}_vs_{engine.name}_level_{level}_1k_e0-e25_eval_edax_level_60.pos')

    #    colors = iter(cm.rainbow(np.linspace(0, 1, 25)))
    #    for e in range(1, 26):
    #        pos_score_filtered = [ps for ps in pos_score if ps.pos.empty_count() == e]
    #        x = [score for score in range(-64, 65, 2)]
    #        y = [sum(1 for ps in pos_score_filtered if ps.score == score) for score in range(-64, 65, 2)]
    #        plt.scatter(x, y, label=f'empty fields = {e}', color=next(colors))
    #    plt.title(f'{engine.fully_qualified_name(level)}')
    #    plt.xlabel('score')
    #    plt.ylabel('number of positions')
    #    plt.show()

    
    #edax = edax.Engine(r'G:\edax-ms-windows\edax-4.4')
    #db = DataBase(r'G:\Reversi\DB.db')

    #db.create_table()
    #db.add_evaluator('Random')
    #db.add_evaluator('Edax4.4')
    #db.add_evaluator('Cassandra')

    ## Add all unique chilren d50 - e60
    #for d in range(11):
    #    name = f'all_unique_e{60-d}'
    #    start = timeit.default_timer()
    #    db.add_group_with_positions(name, AllUniqueChildren(Position.start(), d))
    #    stop = timeit.default_timer()
    #    print(f'Created {name} in {stop - start}.')
    #db.commit()

    #all_unique_e50 = db.get_positions_of_group('all_unique_e50')    

    #add_self_play_to_db(db, random.sample(all_unique_e50, 1_000), RandomEngine(), 0)
    #for level in [0, 5, 10, 15, 20]:
    #    add_self_play_to_db(db, random.sample(all_unique_e50, 1_000), edax, level)

    #all_pos = []
    #all_pos += db.get_positions_of_contrahents('Random', '0', 'Random', '0')
    #for level in [0, 5, 10, 15, 20]:
    #    all_pos += db.get_positions_of_contrahents('Edax4.4', str(level), 'Edax4.4', str(level))

    ## Evaluate positions with edax level 0
    #start = timeit.default_timer()
    #results = edax.solve(all_pos, level=0)
    #stop = timeit.default_timer()
    #for pos, r in zip(all_pos, results):
    #    db.add_evaluation_to_position(pos, 'Edax4.4', r.depth, r.confidence, r.score)
    #print(f'Evaluated positions with edax level 0 in {stop - start}.')
    #db.commit()

    ## Evaluate positions e0 - e25 with edax level 60
    #filtered_pos = [pos for pos in all_pos if pos.EmptyCount() <= 25]
    #start = timeit.default_timer()
    #results = edax.solve(filtered_pos, level=60)
    #stop = timeit.default_timer()
    #for pos, r in zip(filtered_pos, results):
    #    db.add_evaluation_to_position(pos, 'Edax4.4', r.depth, r.confidence, r.score)
    #print(f'Evaluated positions e0 - e25 with edax level 60 in {stop - start}.')
    #db.commit()

    #x, y, yerr = eval_accuracy(db, 'Edax4.4', 'Random', 0, 'Random', 0)
    #plt.errorbar(x, y, yerr, fmt='o', label='Random self play')

    #for level in [0, 5, 10, 15, 20]:
    #    x, y, yerr = eval_accuracy(db, 'Edax4.4', 'Edax4.4', level, 'Edax4.4', level)
    #    plt.errorbar(x, y, yerr, fmt='o', label=f'Edax4.4 self play level {level}')
    #plt.ylim(0, 12)
    #plt.legend(loc=4)
    #plt.show()

    #for empty_count in range(1, 26):
    #    x, y = score_histogram(db, 'Edax4.4', 'Random', 0, 'Random', 0, empty_count)
    #    plt.scatter(x, y, label=f'{empty_count=}')
    #plt.show()

    #for level in [0, 5, 10, 15, 20]:
    #    for empty_count in range(1, 26):
    #        x, y = score_histogram(db, 'Edax4.4', 'Edax4.4', level, 'Edax4.4', level, empty_count)
    #        plt.scatter(x, y, label=f'{empty_count=}')
    #    plt.show()

#Created all_unique_e60 in 0.0007465999806299806.
#Created all_unique_e59 in 0.00191430002450943.
#Created all_unique_e58 in 0.0027142000035382807.
#Created all_unique_e57 in 0.007895099988672882.
#Created all_unique_e56 in 0.030821399996057153.
#Created all_unique_e55 in 0.1594290000211913.
#Created all_unique_e54 in 0.8967608000093605.
#Created all_unique_e53 in 5.264467100001639.
#Created all_unique_e52 in 33s
#Created all_unique_e51 in 227s
#Created all_unique_e50 in 1548s, 26min
#Self played Random level 0 in 29s
#Self played Edax4.4 level 0 in 411s
#Self played Edax4.4 level 5 in 458s
#Self played Edax4.4 level 10 in 1993s, 33min
#Self played Edax4.4 level 15 in 2707s, 45min
#Self played Edax4.4 level 20 in 10'386s, 3h
#Evaluated positions with edax level 0 in 1949s, 30min
#Evaluated positions e0 - e25 with edax level 60 in 17'544s, 5h