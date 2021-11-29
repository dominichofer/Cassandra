cmake_minimum_required(VERSION 3.18)
project(Cassandra LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "-march=native -w -mmmx -msse -msse2 -msse3 -mssse3 -msse4 -msse4a -msse4.1 -msse4.2 -mavx -mavx2")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

include(FetchContent)

# Google Test
FetchContent_Declare(googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.10.x
)
FetchContent_MakeAvailable(googletest)
set(BENCHMARK_ENABLE_GTEST_TESTS "OFF")
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Google Benchmark
FetchContent_Declare(googlebenchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG master
)
FetchContent_MakeAvailable(googlebenchmark)

# Range v3
FetchContent_Declare(range-v3
        GIT_REPOSITORY https://github.com/ericniebler/range-v3.git
        GIT_TAG 0.11.0
)
FetchContent_MakeAvailable(range-v3)
link_libraries(range-v3)

# FMT
FetchContent_Declare(fmt
        GIT_REPOSITORY https://github.com/fmtlib/fmt.git
        GIT_TAG 8.0.1
)
FetchContent_MakeAvailable(fmt)
link_libraries(fmt)

# OpenMP
find_package(OpenMP)
link_libraries(OpenMP::OpenMP_CXX)

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
add_subdirectory(SearchBenchmark)
add_subdirectory(IO)
add_subdirectory(IoTest)

add_subdirectory(PatternFit)
add_subdirectory(OEIS)
add_subdirectory(Perft)