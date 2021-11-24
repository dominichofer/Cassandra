add_executable(
    Perft
    Hashtable.cpp
    Hashtable.h
    Main.cpp
    Perft.cpp
    Perft.h
)

target_link_libraries(Perft LINK_PUBLIC Search IO)