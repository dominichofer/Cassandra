from core import *
import cassandra
import edax
import datetime
import timeit
import math
import statistics
import matplotlib.pyplot as plt


def print_timestamped(text: str = ''):
    print(f'[{datetime.datetime.now()}] {text}')


def edax_engine(level: int):
    return edax.Engine(r'G:\edax-ms-windows\edax-4.4', level=level)


def cassandra_engine(model: str, level: str):
    return cassandra.Engine(
            r'C:\Users\Dominic\source\repos\Cassandra\bin\Solver.exe',
            model,
            level=level)


def SD(n, s):
   return s * math.sqrt(math.e * math.pow(1 - (1 / n), n - 1) - 1)


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


if __name__ == '__main__':
    files = [
        r'G:\Edax4.4_level_0_vs_Edax4.4_level_0_from_e54.gs',
        r'G:\Edax4.4_level_5_vs_Edax4.4_level_5_from_e54.gs',
        r'G:\Edax4.4_level_10_vs_Edax4.4_level_10_from_e54.gs',
        r'G:\Edax4.4_level_15_vs_Edax4.4_level_15_from_e54.gs',
        r'G:\Edax4.4_level_20_vs_Edax4.4_level_20_from_e54.gs',
    ]

    ## clear_scores
    ##for f in files:
    ##    game_scores = parse_game_score_file(f)
    ##    for gs in game_scores:
    ##        gs.clear_scores()
    ##    write_game_score_file(game_scores, f)
    ##quit()
    
    #for f in files:
    #    game_scores = parse_game_score_file(f)
    #    for chunk in chunks(game_scores, 64):
    #        start = timeit.default_timer()
    #        solve_game_scores(edax_engine(60), chunk, 29, 30)
    #        write_game_score_file(game_scores, f)
    #        stop = timeit.default_timer()
    #        print(f'{f} {stop - start}')
    #quit()

    engines = [
        edax_engine(0),
        cassandra_engine(r'G:\Reversi2\iteration1.model', 0),
        cassandra_engine(r'G:\Reversi2\iteration2.model', 0),
        ]
    names = [
        'Edax4.4 level 0',
        'Ours Iteration 1',
        'Ours Iteration 2',
        ]

    #print_timestamped('begin')

    #files = [
    #    #r'G:\Reversi2\Random_vs_Random_from_e54.gs',
    #    r'G:\Reversi2\Edax4.4_level_0_vs_Edax4.4_level_0_from_e54.gs',
    #    r'G:\Reversi2\Edax4.4_level_5_vs_Edax4.4_level_5_from_e54.gs',
    #    r'G:\Reversi2\Edax4.4_level_10_vs_Edax4.4_level_10_from_e54.gs',
    #    r'G:\Reversi2\Edax4.4_level_15_vs_Edax4.4_level_15_from_e54.gs',
    #    r'G:\Reversi2\Edax4.4_level_20_vs_Edax4.4_level_20_from_e54.gs',
    #]
    pos_scores = [
        (pos, score)
        for file in files
        for game_score in parse_game_score_file(file)
        for pos, score in game_score.pos_scores()
        if score != undefined_score
    ]
    pos, exact_scores = zip(*pos_scores)
    
    #print_timestamped('evaluated')
    
    for eng, name in zip(engines, names):
        scores = [line.score for line in eng.solve(pos)]
        x = []
        stdev = []
        stdev_err = []
        for e in range(61):
            diff = [a - b for a, b, p in zip(exact_scores, scores, pos) if p.empty_count() == e]
            if len(diff) > 1:
                x.append(e)
                s = statistics.stdev(diff)
                stdev.append(s)
                stdev_err.append(SD(len(diff), s))
        #plt.errorbar(x, stdev, stdev_err, fmt='.', label=name)
        plt.plot(x, stdev, label=name)
        plt.fill_between(x, [s - 2 * e for s, e in zip(stdev, stdev_err)], [s + 2 * e for s, e in zip(stdev, stdev_err)], alpha=.5)
    
    plt.xlabel('empty fields')
    plt.ylabel('standard deviation')
    #plt.xlim(0, 27)
    #plt.ylim(0, 7)
    plt.grid(True, which='major', axis='both')
    plt.legend(loc=4)
    plt.savefig(r'G:\Reversi2\SD_all.png')
    plt.show()
    #plt.clf()
    
    #print_timestamped('end')
    #quit()

    #cass = cassandra.Engine(
    #    r'C:\Users\Dominic\source\repos\Cassandra\bin\Solver.exe',
    #    r'G:\Reversi2\pattern_edax_bs_5_spd5_train_100000_d5_exact0_accfit_0_0_0_it5.model',
    #    level=0)
    #start_pos = parse_position_file(r'G:\Reversi2\all_unique_e54.script')
    #games = [Game(p) for p in start_pos]
    
    #for d in [0, 5, 10, 15, 20]:
    #    engine = cassandra.Engine(
    #        r'C:\Users\Dominic\source\repos\Cassandra\bin\Solver.exe',
    #        r'G:\Reversi2\pattern_edax_bs_5_spd5_train_100000_d5_exact0_accfit_0_0_0_it0.model',
    #        level=d)
    #    name = engine.name('_')

    #    game_scores = [GameScore.from_game(g) for g in self_play(engine, games)]
    #    print_timestamped(f'self_play complete {d}')
        
    #    solve_game_scores(edax_engine(60), game_scores, 0, 25)
    #    print_timestamped(f'solve complete {d}')

    #    write_game_score_file(game_scores, rf'G:\Reversi2\it0_{name}_vs_{name}_from_e54.gs')
    #quit()

    #games = parse_game_file(r'selfplay_d0_pattern_edax_bs_5_spd5_train_100000_d5_exact3_accfit_0_0_0_it19.game')
    #pos = [p for g in games for p in g.positions()]
    #write_position_file(pos, fr'G:\Reversi2\{name}_vs_{name}_frome54.pos')
    #exit()

    engines = [
        edax_engine(0),
        edax_engine(5),
        edax_engine(10),
        edax_engine(15),
        edax_engine(20),
        ]
    
    #for engine in engines:
    #    name = engine.name("_")
    #    game_scores = parse_game_score_file(fr'G:\{name}_vs_{name}_frome54.gs')
    #    for gs in game_scores:
    #        gs.clear_scores()
    #    write_game_score_file(game_scores, fr'G:\{name}_vs_{name}_frome54.gs')
    #quit()
    for engine in engines:
        name = engine.name("_")

        #start_pos = parse_position_file(r'G:\Reversi2\all_unique_e54.script')
        #games = [Game(p) for p in start_pos]
        
        #start = timeit.default_timer()
        #game_scores = [GameScore.from_game(g) for g in self_play(engine, games)]
        #stop = timeit.default_timer()
        #print(f'self play {name}: {stop - start}')

        game_scores = parse_game_score_file(fr'G:\{name}_vs_{name}_frome54.gs')

        start = timeit.default_timer()
        solve_game_scores(edax_engine(60), game_scores, 0, 20)
        stop = timeit.default_timer()
        print(f'{stop - start}')

        write_game_score_file(game_scores, fr'G:\{name}_vs_{name}_frome54.gs')
        
        #pos = [p for g in games for p in g.positions()]
        #write_position_file(pos, fr'G:\{name}_vs_{name}_frome54.pos')

        #pos_filtered = [p for p in pos if 1 <= p.empty_count() <= 25]

        #start = timeit.default_timer()
        #lines = edax_engine(60).solve(pos_filtered)
        #pos_scores = [PositionScore(pos, line.score) for pos, line in zip(pos_filtered, lines)]
        #stop = timeit.default_timer()
        #print(f'edax level 60: {stop - start}')

        #write_position_score_file(pos_scores, fr'G:\Reversi2\{name}_vs_{name}_frome54_e0-e25_eval_edax_level_60.pos')

#self play Random level=None: 5.348081299802288
#edax level 60: 6856.039577499963
#self play Edax4.4_level_0 level=0: 719.625712600071
#edax level 60: 6658.3153752998915
#self play Edax4.4_level_5 level=5: 682.4868249997962
#edax level 60: 1933.2271673001815
#self play Edax4.4_level_10 level=10: 1735.4956577999983
#edax level 60: 1894.221424800111
#self play Edax4.4_level_15 level=15: 1998.0058520000894
#edax level 60: 2022.0838329999242
#self play Edax4.4_level_20 level=20: 8086.092103400035
#edax level 60: 1973.5147937999573