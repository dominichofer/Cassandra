add_library(
    IO STATIC
    Database.h
    FForum.h
    File.h
    Integers.cpp
    Integers.h
    IO.cpp
    IO.h
    PatternEval.cpp
    PatternEval.h
    PosScore.h
)

target_link_libraries(IO LINK_PUBLIC Core Pattern Search)