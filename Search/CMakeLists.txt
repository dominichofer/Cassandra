add_library(
    Search STATIC
    Algorithm.cpp
    Algorithm.h
    AlphaBetaFailHard.cpp
    AlphaBetaFailSoft.cpp
    Executor.h
    HashTablePVS.cpp
    HashTablePVS.h
    IDAB.cpp
    NegaMax.cpp
    Objects.cpp
    Objects.h
    Player.cpp
    Player.h
    PrincipalVariation.cpp
    Puzzle.cpp
    Puzzle.h
    Search.h
    SortedMoves.h
)

target_link_libraries(Search LINK_PUBLIC Core)