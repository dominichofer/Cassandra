add_executable(
    CoreBenchmark
    main.cpp
    PositionGenerator.cpp
)

target_link_libraries(CoreBenchmark LINK_PUBLIC Core Pattern benchmark::benchmark)