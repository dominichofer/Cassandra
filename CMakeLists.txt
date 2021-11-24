cmake_minimum_required(VERSION 3.18)
project(Cassandra LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "-march=native -w -mmmx -msse -msse2 -msse3 -mssse3 -msse4 -msse4a -msse4.1 -msse4.2 -mavx -mavx2")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(BENCHMARK_ENABLE_GTEST_TESTS "OFF")

add_subdirectory(range-v3)
include_directories(range-v3/include)

# add_subdirectory(benchmark)
# include_directories(benchmark/include)

include(FetchContent)

FetchContent_Declare(googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.10.x)

FetchContent_Declare(googlebenchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG master) # need master for benchmark::benchmark

FetchContent_MakeAvailable(
        googletest
        googlebenchmark)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

include_directories(.)
add_subdirectory(Core)
add_subdirectory(CoreTest)
add_subdirectory(CoreBenchmark)
add_subdirectory(Math)
add_subdirectory(MathTest)
add_subdirectory(Pattern)
add_subdirectory(PatternTest)
add_subdirectory(Search)
add_subdirectory(SearchTest)
add_subdirectory(IO)
add_subdirectory(IoTest)
add_subdirectory(OEIS)
add_subdirectory(Perft)