set(CMAKE_CXX_STANDARD 20)

add_library(Math Algorithm.cpp Algorithm.h DenseMatrix.h Math.h Matrix.h MatrixCSR.h Solver.h Statistics.h SymExp.cpp SymExp.h Vector.h)

target_include_directories (Math PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})