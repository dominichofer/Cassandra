set(CMAKE_CXX_STANDARD 20)

add_executable(oeis Main.cpp)

target_link_libraries(oeis LINK_PUBLIC Core Math IO)