add_executable(
    SearchBenchmark
    main.cpp
)

target_link_libraries(SearchBenchmark LINK_PUBLIC Core Pattern Search IO benchmark::benchmark)