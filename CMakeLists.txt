cmake_minimum_required(VERSION 3.18)
project(Cassandra)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "-march=native -w -mmmx -msse -msse2 -msse3 -mssse3 -msse4 -msse4a -msse4.1 -msse4.2 -mavx -mavx2")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

include_directories(range-v3/include)
add_subdirectory(Core)
add_subdirectory(Math)
add_subdirectory(IO)
add_subdirectory(OEIS)