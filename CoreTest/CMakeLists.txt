add_executable(
    CoreTest
    BitBoard.cpp
    Bit.cpp
    CountLastFlip.cpp
    EvalGameOver.cpp
    Flips.cpp
    Metronome.cpp
    MoreTypes.cpp
    Moves.cpp
    pch.cpp
    pch.h
    PositionGenerator.cpp
    PossibleMoves.cpp
    Stability.cpp
)

target_link_libraries(
    CoreTest
    Core
    gtest_main
)

include(GoogleTest)
gtest_discover_tests(CoreTest)