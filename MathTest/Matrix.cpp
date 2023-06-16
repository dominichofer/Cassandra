#include "pch.h"
#include "Matrix.h"

Matrix CreateMatrix2x2(double a, double b, double c, double d)
{
    Matrix m(2, 2);
    m(0, 0) = a;
    m(0, 1) = b;
    m(1, 0) = c;
    m(1, 1) = d;
    return m;
}

Matrix CreateMatrix2x3(double a, double b, double c, double d, double e, double f)
{
    Matrix m(2, 3);
    m(0, 0) = a;
    m(0, 1) = b;
    m(0, 2) = c;
    m(1, 0) = d;
    m(1, 1) = e;
    m(1, 2) = f;
    return m;
}

Matrix CreateMatrix3x2(double a, double b, double c, double d, double e, double f)
{
    Matrix m(3, 2);
    m(0, 0) = a;
    m(0, 1) = b;
    m(1, 0) = c;
    m(1, 1) = d;
    m(2, 0) = e;
    m(2, 1) = f;
    return m;
}

Matrix CreateMatrix3x3(double a, double b, double c, double d, double e, double f, double g, double h, double i)
{
    Matrix m(3, 3);
    m(0, 0) = a;
    m(0, 1) = b;
    m(0, 2) = c;
    m(1, 0) = d;
    m(1, 1) = e;
    m(1, 2) = f;
    m(2, 0) = g;
    m(2, 1) = h;
    m(2, 2) = i;
    return m;
}

TEST(MatrixTest, Constructor)
{
    Matrix matrix(3, 4);

    EXPECT_EQ(matrix.Rows(), 3);
    EXPECT_EQ(matrix.Cols(), 4);

    for (std::size_t i = 0; i < matrix.Rows(); i++)
        for (std::size_t j = 0; j < matrix.Cols(); j++)
            EXPECT_EQ(matrix(i, j), 0.0);
}

TEST(MatrixTest, IdentityMatrix)
{
    Matrix identity = Matrix::Id(3);

    EXPECT_EQ(identity.Rows(), 3);
    EXPECT_EQ(identity.Cols(), 3);

    for (std::size_t i = 0; i < identity.Rows(); i++)
        for (std::size_t j = 0; j < identity.Cols(); j++)
            if (i == j)
                EXPECT_EQ(identity(i, j), 1.0);
            else
                EXPECT_EQ(identity(i, j), 0.0);
}

TEST(MatrixTest, Addition)
{
    Matrix matrix1 = CreateMatrix2x2(1, 2, 3, 4);
    Matrix matrix2 = CreateMatrix2x2(5, 6, 7, 8);

    Matrix sum = matrix1 + matrix2;

    EXPECT_EQ(sum, CreateMatrix2x2(6, 8, 10, 12));
}

TEST(MatrixTest, Subtraction)
{
    Matrix matrix1 = CreateMatrix2x2(4, 3, 2, 1);
    Matrix matrix2 = CreateMatrix2x2(1, 2, 3, 4);

    Matrix difference = matrix1 - matrix2;

    EXPECT_EQ(difference, CreateMatrix2x2(3, 1, -1, -3));
}

TEST(MatrixTest, ScalarMultiplication)
{
    Matrix matrix = CreateMatrix2x2(1, 2, 3, 4);

    Matrix scaled = matrix * 2.0;

    EXPECT_EQ(scaled, CreateMatrix2x2(2, 4, 6, 8));
}

TEST(MatrixTest, ScalarDivision)
{
    Matrix matrix = CreateMatrix2x2(6, 8, 10, 12);

    Matrix divided = matrix / 2.0;

    EXPECT_EQ(divided, CreateMatrix2x2(3, 4, 5, 6));
}

TEST(MatrixTest, MatrixVectorMultiplication)
{
    Matrix matrix = CreateMatrix2x3(1, 2, 3, 4, 5, 6);
    Vector vector(std::vector{7.0, 8.0, 9.0});

    Vector product = matrix * vector;

    EXPECT_EQ(product.size(), 2);
    EXPECT_EQ(product[0], 50.0);
    EXPECT_EQ(product[1], 122.0);
}

TEST(MatrixTest, VectorMatrixMultiplication)
{
    Matrix matrix = CreateMatrix3x2(4, 5, 6, 7, 8, 9);
    Vector vector(std::vector{1.0, 2.0, 3.0});

    Vector product = vector * matrix;

    EXPECT_EQ(product.size(), 2);
    EXPECT_EQ(product[0], 40.0);
    EXPECT_EQ(product[1], 46.0);
}

TEST(MatrixTest, MatrixMultiplication)
{
    Matrix matrix1 = CreateMatrix2x3(1, 2, 3, 4, 5, 6);
    Matrix matrix2 = CreateMatrix3x2(7, 8, 9, 10, 11, 12);

    Matrix product = matrix1 * matrix2;

    EXPECT_EQ(product, CreateMatrix2x2(58, 64, 139, 154));
}

TEST(MatrixTest, Transposition)
{
    Matrix matrix = CreateMatrix2x3(1, 2, 3, 4, 5, 6);

    Matrix transposedMatrix = transposed(matrix);

    EXPECT_EQ(transposedMatrix, CreateMatrix3x2(1, 4, 2, 5, 3, 6));
}

TEST(MatrixTest, ToString)
{
    Matrix matrix = CreateMatrix2x2(1, 2, 3, 4);

    std::string expected = "{{1.000000 2.000000}\n{3.000000 4.000000}}";
    std::string result = to_string(matrix);

    EXPECT_EQ(result, expected);
}
