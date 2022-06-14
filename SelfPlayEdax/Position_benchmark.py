import timeit
import numpy as np
from Position import *

# Decorator
def Benchmark(function):
    start = timeit.default_timer()
    function()
    stop = timeit.default_timer()
    diff = (stop - start) * 1_000_000
    name = function.__name__
    if name.startswith('benchmark_'):
        name = name[len('benchmark_'):]
    print(f'{name}: {diff:0.0f} us')

@Benchmark
def benchmark_popcount():
    popcount(np.uint64(36))
        
@Benchmark
def benchmark_FirstSetField():
    FirstSetField(np.uint64(36))
    
@Benchmark
def benchmark_FirstSetCleared():
    FirstSetCleared(np.uint64(36))
    
@Benchmark
def benchmark_FlipCodiagonal():
    FlipCodiagonal(Position.Start())
    
@Benchmark
def benchmark_FlipDiagonal():
    FlipDiagonal(Position.Start())
    
@Benchmark
def benchmark_FlipHorizontal():
    FlipHorizontal(Position.Start())
    
@Benchmark
def benchmark_FlipVertical():
    FlipVertical(Position.Start())
    
@Benchmark
def benchmark_FlipToUnique():
    FlipToUnique(Position.Start())
    
@Benchmark
def benchmark_PossibleMoves():
    PossibleMoves(Position.Start())

def ChildrenBenchmark(empty_count_diff):
    pos = Position.Start()
    start = timeit.default_timer()
    count = sum(1 for _ in Children(pos, empty_count_diff))
    stop = timeit.default_timer()
    diff = count / (stop - start)
    print(f'Children({empty_count_diff}): {diff} children/s')

def AllUniqueChildrenBenchmark(empty_count_diff):
    pos = Position.Start()
    start = timeit.default_timer()
    count = sum(1 for _ in AllUniqueChildren(pos, empty_count_diff))
    stop = timeit.default_timer()
    diff = count / (stop - start)
    print(f'AllUniqueChildren({empty_count_diff}): {diff} children/s')
    
if __name__ == '__main__':
    for i in range(3, 7):
        ChildrenBenchmark(i)
    for i in range(3, 7):
        AllUniqueChildrenBenchmark(i)