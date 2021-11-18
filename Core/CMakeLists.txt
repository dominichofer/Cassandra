set(CMAKE_CXX_STANDARD 20)

add_library(Core Algorithms.h BitBoard.cpp BitBoard.h Bit.h Confidence.h Core.h CountLastFlip.cpp Flips.cpp HashTable.h HasMoves.cpp Interval.cpp Interval.h Metronome.cpp Metronome.h MoreTypes.h Moves.h Player.cpp Player.h Position.cpp PositionGenerator.cpp PositionGenerator.h Position.h PossibleMoves.cpp Ranges.h Stability.cpp)

target_include_directories (Core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})