add_executable(
    PatternTest
    pch.cpp
    pch.h
    test.cpp
)

target_link_libraries(
    PatternTest
    Pattern
    gtest_main
)

include(GoogleTest)
gtest_discover_tests(PatternTest)