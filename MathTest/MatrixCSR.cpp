#include "pch.h"

MatrixCSR CreateMatrix()
{
    // 1 1 0 0
    // 0 1 1 0
    // 0 0 0 2

    MatrixCSR m(2, 3, 4);
    m.Row(0)[0] = 0;
    m.Row(0)[1] = 1;
    m.Row(1)[0] = 1;
    m.Row(1)[1] = 2;
    m.Row(2)[0] = 3;
    m.Row(2)[1] = 3;
    return m;
}

TEST(MatrixCSRTest, RowsAndCols)
{
    MatrixCSR matrix = CreateMatrix();

    EXPECT_EQ(matrix.Rows(), 3);
    EXPECT_EQ(matrix.Cols(), 4);
    EXPECT_EQ(matrix.Row(0).size(), 2);
}

TEST(MatrixCSRTest, MatrixVectorMultiplication)
{
    MatrixCSR matrix = CreateMatrix();
    Vector vector({ 1, 2, 3, 4 });

    Vector result = matrix * vector;

    EXPECT_EQ(result, Vector({ 3, 5, 8 }));
}

TEST(MatrixCSRTest, VectorMatrixMultiplication)
{
    MatrixCSR matrix = CreateMatrix();
    Vector vector({ 1, 2, 3 });

    Vector result = vector * matrix;

    EXPECT_EQ(result, Vector({ 1, 3, 2, 6 }));
}

TEST(MatrixCSRTest, JacobiPreconditionerOfATA)
{
    MatrixCSR matrix = CreateMatrix();

    Vector result = JacobiPreconditionerOfATA(matrix);

    EXPECT_EQ(result, Vector({ 1, 0.5, 1, 0.25 }));
}