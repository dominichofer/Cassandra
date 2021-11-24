add_executable(
    SearchTest
    Interval.cpp
    Objects.cpp
    pch.cpp
    pch.h
    Searches.cpp
    StatusQuo.cpp
)

target_link_libraries(
    SearchTest
    Core
    Search
    gtest_main
)

include(GoogleTest)
gtest_discover_tests(SearchTest)