add_executable(
    IoTest
    pch.cpp
    pch.h
    test.cpp
)

target_link_libraries(
    IoTest
    IO
    gtest_main
)

include(GoogleTest)
gtest_discover_tests(IoTest)