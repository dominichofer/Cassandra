#include "pch.h"
#include "DenseMatrix.h"

TEST(DenseMatrix, equality)
{
	EXPECT_TRUE(Matrix(1, 2, 3, 4) == Matrix(1, 2, 3, 4));
	EXPECT_FALSE(Matrix(1, 2, 3, 4) != Matrix(1, 2, 3, 4));
	EXPECT_FALSE(Matrix(1, 2, 3, 4) == Matrix(1, 2, 3, 5));
	EXPECT_TRUE(Matrix(1, 2, 3, 4) != Matrix(1, 2, 3, 5));
	EXPECT_TRUE(Matrix(1.0, 2.0, 3.0, 4.0) == Matrix(1.0, 2.0, 3.0, 4.0));
	EXPECT_FALSE(Matrix(1.0, 2.0, 3.0, 4.0) != Matrix(1.0, 2.0, 3.0, 4.0));
	EXPECT_FALSE(Matrix(1.0, 2.0, 3.0, 4.0) == Matrix(1.0, 2.0, 3.0, 5.0));
	EXPECT_TRUE(Matrix(1.0, 2.0, 3.0, 4.0) != Matrix(1.0, 2.0, 3.0, 5.0));
}

TEST(DenseMatrix, add)
{
	DenseMatrix a = Matrix(1, 2, 3, 4);
	DenseMatrix b = Matrix(2, 4, 6, 8);
	EXPECT_TRUE(a + a == b);
	a += a;
	EXPECT_TRUE(a == b);
}

TEST(DenseMatrix, sub)
{
	DenseMatrix a = Matrix(1, 2, 3, 4);
	DenseMatrix b = Matrix(0, 0, 0, 0);
	EXPECT_TRUE(a - a == b);
	a -= a;
	EXPECT_TRUE(a == b);
}

TEST(DenseMatrix, sub_special)
{
	DenseMatrix a = Matrix(1, 2, 3, 4);
	DenseMatrix b = Matrix(1, 2, 3, 4);
	DenseMatrix c = Matrix(0, 0, 0, 0);
	EXPECT_TRUE(a - std::move(b) == c);
}

TEST(DenseMatrix, mul_value)
{
	DenseMatrix a = Matrix(1, 2, 3, 4);
	DenseMatrix b = Matrix(2, 4, 6, 8);
	EXPECT_TRUE(a * 2 == b);
	EXPECT_TRUE(2 * a == b);
	a *= 2;
	EXPECT_TRUE(a == b);
}

TEST(DenseMatrix, div_value)
{
	DenseMatrix a = Matrix(1, 2, 3, 4);
	DenseMatrix b = Matrix(2, 4, 6, 8);
	EXPECT_TRUE(b / 2 == a);
	b /= 2;
	EXPECT_TRUE(a == b);
}

TEST(DenseMatrix, mul_vec)
{
	DenseMatrix A = Matrix(1, 2, 3, 4);
	std::valarray x = { 1, 2 };
	std::valarray b = { 5, 11 };
	EXPECT_TRUE(AllTrue(A * x == b));
}

TEST(DenseMatrix, mul_vec2)
{
	DenseMatrix A = Matrix(1, 2, 3, 4);
	std::valarray x = { 1, 2 };
	std::valarray b = { 7, 10 };
	EXPECT_TRUE(AllTrue(x * A == b));
}

TEST(DenseMatrix, mul_mat)
{
	DenseMatrix A = Matrix(1, 2, 3, 4);
	DenseMatrix B = Matrix(5, 6, 7, 8);
	DenseMatrix C = Matrix(1 * 5 + 2 * 7, 1 * 6 + 2 * 8, 3 * 5 + 4 * 7, 3 * 6 + 4 * 8);
	EXPECT_TRUE(A * B == C);
}

TEST(DenseMatrix, transposed)
{
	DenseMatrix a = Matrix(1, 2, 3, 4);
	DenseMatrix b = Matrix(1, 3, 2, 4);
	EXPECT_TRUE(transposed(a) == b);
}
