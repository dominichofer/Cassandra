set(CMAKE_CXX_STANDARD 20)

add_library(IO Database.h FForum.h File.h Integers.cpp Integers.h IO.cpp IO.h PatternEval.cpp PatternEval.h PosScore.h)

target_link_libraries(IO LINK_PUBLIC Core)
target_include_directories(IO PUBLIC ${CMAKE_SOURCE_DIR})