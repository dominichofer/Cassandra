add_executable(
    MathTest
    pch.cpp
    pch.h
    test.cpp
)

target_link_libraries(
    MathTest
    Math
    gtest_main
)

include(GoogleTest)
gtest_discover_tests(MathTest)