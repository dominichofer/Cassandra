add_library(
    Pattern STATIC
    DenseIndexer.cpp
    DenseIndexer.h
    Evaluator.cpp
    Evaluator.h
    Helpers.cpp
    Helpers.h
)

target_link_libraries(Pattern LINK_PUBLIC Core)