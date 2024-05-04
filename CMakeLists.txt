cmake_minimum_required(VERSION 3.18)
project(Cassandra LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
if (MSVC)
    set(CMAKE_CXX_FLAGS "/arch:AVX2 /std:c++latest /EHsc /MP")
    set(CMAKE_CXX_FLAGS_DEBUG "/Od /MDd")
    set(CMAKE_CXX_FLAGS_RELEASE "/Ox /MD /GL")
else()
	set(CMAKE_CXX_FLAGS "-march=native -w -mmmx -msse -msse2 -msse3 -mssse3 -msse4 -msse4a -msse4.1 -msse4.2 -mavx -mavx2")
	set(CMAKE_CXX_FLAGS_DEBUG "-g")
	set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
endif()

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

# OpenMP
find_package(OpenMP)
link_libraries(OpenMP::OpenMP_CXX)

#grpc
FetchContent_Declare(
  gRPC
  GIT_REPOSITORY https://github.com/grpc/grpc
  GIT_TAG        v1.59.3
)
set(FETCHCONTENT_QUIET OFF)
FetchContent_MakeAvailable(gRPC)

include_directories(.)
add_subdirectory(Base)
add_subdirectory(BaseTest)
add_subdirectory(BaseBenchmark)
add_subdirectory(Board)
add_subdirectory(BoardTest)
add_subdirectory(BoardBenchmark)
add_subdirectory(Game)
add_subdirectory(GameTest)
add_subdirectory(GameBenchmark)
add_subdirectory(Search)
add_subdirectory(SearchTest)
add_subdirectory(SearchBenchmark)
add_subdirectory(Math)
add_subdirectory(MathTest)
add_subdirectory(MathBenchmark)
add_subdirectory(Pattern)
add_subdirectory(PatternTest)
add_subdirectory(PatternBenchmark)
add_subdirectory(PatternFit)
add_subdirectory(IO)
add_subdirectory(IoTest)
add_subdirectory(PerftCuda)

add_subdirectory(OEIS)
add_subdirectory(Perft)
