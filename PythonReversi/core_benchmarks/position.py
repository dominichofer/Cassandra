import timeit
from numpy import uint64
from core import Position, play, play_pass, possible_moves

# Decorator
def Benchmark(function):
    start = timeit.default_timer()
    for _ in range(10_000):
        function()
    stop = timeit.default_timer()
    diff = (stop - start) * 100
    name = function.__name__
    if name.startswith('benchmark_'):
        name = name[len('benchmark_'):]
    print(f'{name}: {diff:0.0f} us')
    

@Benchmark
def benchmark_possible_moves():
    possible_moves(Position.start())


#@Benchmark
#def benchmark_flipped_codiagonal():
#    flipped_codiagonal(Position.start())
    

#@Benchmark
#def benchmark_flipped_diagonal():
#    flipped_diagonal(Position.start())
    

#@Benchmark
#def benchmark_flipped_horizontal():
#    flipped_horizontal(Position.start())
    

#@Benchmark
#def benchmark_flipped_vertical():
#    flipped_vertical(Position.start())
    

#@Benchmark
#def benchmark_flipped_to_unique():
#    flipped_to_unique(Position.start())


#def ChildrenBenchmark(empty_count_diff):
#    pos = Position.start()
#    start = timeit.default_timer()
#    count = sum(1 for _ in Children(pos, empty_count_diff))
#    stop = timeit.default_timer()
#    diff = count / (stop - start)
#    print(f'Children({empty_count_diff}): {diff:.0f} children/s')


#def AllUniqueChildrenBenchmark(empty_count_diff):
#    pos = Position.start()
#    start = timeit.default_timer()
#    count = sum(1 for _ in AllUniqueChildren(pos, empty_count_diff))
#    stop = timeit.default_timer()
#    diff = count / (stop - start)
#    print(f'AllUniqueChildren({empty_count_diff}): {diff:.0f} children/s')


#def main():
#    for i in range(3, 7):
#        ChildrenBenchmark(i)
#    for i in range(3, 7):
#        AllUniqueChildrenBenchmark(i)
   

#if __name__ == '__main__':
    #main()