from core import *
import cassandra
import edax
import datetime
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


if __name__ == '__main__':
    engines = [
        edax_engine(0),
        cassandra_engine(r'G:\Reversi2\iteration1.model', 0),
        cassandra_engine(r'G:\Reversi2\iteration2.model', 0),
        cassandra_engine(r'G:\Reversi2\iteration3.model', 0),
        cassandra_engine(r'G:\Reversi2\iteration4.model', 0),
        cassandra_engine(r'G:\Reversi2\iteration5.model', 0),
        cassandra_engine(r'G:\Reversi2\iteration6.model', 0),
        cassandra_engine(r'G:\Reversi2\iteration7.model', 0),
        cassandra_engine(r'G:\Reversi2\iteration8.model', 0),
        cassandra_engine(r'G:\Reversi2\iteration9.model', 0),
        cassandra_engine(r'G:\Reversi2\iteration10.model', 0),
        #cassandra_engine(r'G:\Reversi2\it1.model', 0),
        #cassandra_engine(r'G:\Reversi2\it2.model', 0),
        #cassandra_engine(r'G:\Reversi2\it3.model', 0),
        #cassandra_engine(r'G:\Reversi2\blocksize5_it100.model', 0),
        #cassandra_engine(r'G:\Reversi2\pattern_edax_bs_5_spd5_train_100000_d5_exact3_accfit_0_0_0_it0.model', 0),
        #cassandra_engine(r'G:\Reversi2\pattern_edax_bs_5_spd5_train_100000_d5_exact3_accfit_0_0_0_it1.model', 0),
        ]
    names = [
        'Edax4.4 level 0',
        'Iteration 1',
        'Iteration 2',
        'Iteration 3',
        'Iteration 4',
        'Iteration 5',
        'Iteration 6',
        'Iteration 7',
        'Iteration 8',
        'Iteration 9',
        'Iteration 10',
        ]

    print_timestamped('begin')

    files = [
        #r'G:\Reversi2\Random_vs_Random_from_e54.gs',
        r'G:\Reversi2\Edax4.4_level_0_vs_Edax4.4_level_0_from_e54.gs',
        r'G:\Reversi2\Edax4.4_level_5_vs_Edax4.4_level_5_from_e54.gs',
        r'G:\Reversi2\Edax4.4_level_10_vs_Edax4.4_level_10_from_e54.gs',
        r'G:\Reversi2\Edax4.4_level_15_vs_Edax4.4_level_15_from_e54.gs',
        r'G:\Reversi2\Edax4.4_level_20_vs_Edax4.4_level_20_from_e54.gs',
    ]
    pos_scores = [
        (pos, score)
        for file in files
        for game_score in parse_game_score_file(file)
        for pos, score in game_score.pos_scores()
        if score != undefined_score
    ]
    pos, exact_scores = zip(*pos_scores)
    
    print_timestamped('evaluated')
    
    for eng, name in zip(engines, names):
        scores = [line.score for line in eng.solve(pos)]
        x = []
        stdev = []
        stdev_err = []
        full_diff = []
        for e in range(1, 29):
            diff = [a - b for a, b, p in zip(exact_scores, scores, pos) if p.empty_count() == e and a != undefined_score]
            full_diff += diff
            if len(diff) > 1:
                x.append(e)
                s = statistics.stdev(diff)
                stdev.append(s)
                stdev_err.append(SD(len(diff), s))
        plt.plot(x, stdev, label=name)
        plt.fill_between(x, [s - 2 * e for s, e in zip(stdev, stdev_err)], [s + 2 * e for s, e in zip(stdev, stdev_err)], alpha=.25)
        mad = statistics.mean(abs(d) for d in diff)
        print(f'{eng.name} stdev:{statistics.stdev(diff)} mad:{mad}')
    
    plt.xlabel('empty fields')
    plt.ylabel('standard deviation')
    #plt.xlim(0, 27)
    #plt.ylim(0, 10)
    plt.grid(True, which='major', axis='both')
    plt.legend(loc=4)
    plt.savefig(r'G:\Reversi2\SD2.png')
    plt.show()
    
    print_timestamped('end')