add_library(
    PatternFit
    PatternFit.cpp
    PatternFit.h
)

target_link_libraries(PatternFit LINK_PUBLIC Math Search Pattern IO)