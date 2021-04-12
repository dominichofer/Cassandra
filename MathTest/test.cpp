#include "pch.h"
#include <random>

TEST(Statistics, Avg)
{
	EXPECT_EQ(Average(std::vector{1}), 1);
	EXPECT_EQ(Average(std::vector{1,1}), 1);
	EXPECT_EQ(Average(std::vector{1,2}), 1.5);
}
TEST(Statistics, Var)
{
	EXPECT_EQ(Variance(std::vector{1}), 0);
	EXPECT_EQ(Variance(std::vector{1,1}), 0);
	EXPECT_EQ(Variance(std::vector{1,2}), 0.25);
}

TEST(DenseMatrix, mult)
{
	DenseMatrix A(4, 3);
	for (int i = 0; i < A.Rows(); i++)
		for (int j = 0; j < A.Cols(); j++)
			A(i, j) = i * 10 + j;

	Vector x = {1,2,3};
	Vector b = {0,0,0,0};
	for (int i = 0; i < A.Rows(); i++)
		for (int j = 0; j < A.Cols(); j++)
			b[i] += A(i, j) * x[j];

	EXPECT_TRUE(A * x == b);
}

TEST(MatrixCSR, mult)
{
	MatrixCSR<int> A(1, 3, 4);
	DenseMatrix AA(4, 3);
	*(A.begin() + 0) = 0; AA(0,0) = 1;
	*(A.begin() + 1) = 2; AA(1,2) = 1;
	*(A.begin() + 2) = 1; AA(2,1) = 1;
	*(A.begin() + 3) = 0; AA(3,0) = 1;

	Vector x = {1,2,3};
	EXPECT_TRUE(A.Ax(x) == AA.Ax(x));
	EXPECT_TRUE(A * x == AA * x);

	x = {1,2,3,4};
	EXPECT_TRUE(transposed(A) * x == transposed(AA) * x);
	EXPECT_TRUE(A.ATx(x) == AA.ATx(x));
}

TEST(CG, DenseMatrix)
{
	const std::size_t size = 4;
	DenseMatrix A(size, size);
	Vector x = {1,2,3,4};
	Vector x0(size);

	std::mt19937_64 rnd_engine(13);
	auto rnd = [&rnd_engine]() { return std::uniform_real_distribution<float>(0, 1)(rnd_engine); };

	for (int i = 0; i < A.Rows(); i++)
		for (int j = 0; j < A.Cols(); j++)
			A(i, j) = rnd();
	A += transposed(A) + static_cast<float>(size) * DenseMatrix::Id(size); // makes it positive definite

	Vector b = A * x;

	CG cg(A, b, x0);
	cg.Iterate(size);

	EXPECT_LT(cg.Residuum(), 1e-8);
	EXPECT_LT(norm(cg.Error()), 1e-4);
}

//TEST(CG, MatCSR_Vec)
//{
//	const std::size_t size = 4;
//	Matrix<float> B(size, size);
//	MatrixCSR<float, uint8_t> A(size);
//	Vector x(size);
//	Vector _0(size);
//
//	std::mt19937_64 rnd_engine(13);
//	auto rnd = [&rnd_engine]() { return std::uniform_real_distribution<float>(0, 1)(rnd_engine); };
//
//	for (int i = 0; i < B.Rows(); i++)
//		for (int j = 0; j < B.Cols(); j++)
//			B(i, j) = rnd();
//	B += Transposed(B) + size * Matrix<float>::Id(size); // makes it positive-definite
//
//	for (int i = 0; i < B.Rows(); i++)
//	{
//		for (int j = 0; j < B.Cols(); j++)
//			if ((i + j) % 2 == 0)
//				A.push_back(j, B(i, j));
//		A.end_row();
//	}
//
//	for (int i = 0; i < x.size(); i++)
//		x[i] = i;
//
//	Vector b = A * x;
//
//	CG solver(A, b, _0);
//	solver.Iterate(size);
//
//	EXPECT_LT(solver.Residuum(), 1e-8);
//	EXPECT_LT(norm(solver.Error()), 1e-6);
//}

//TEST(LSQR, MatCSR_Vec)
//{
//	const std::size_t size = 4;
//	Matrix<float> B(size, size);
//	MatrixCSR<float, uint8_t> A(size);
//	Vector x(size);
//	Vector _0(size);
//
//	std::mt19937_64 rnd_engine(13);
//	auto rnd = [&rnd_engine]() { return std::uniform_real_distribution<float>(0, 1)(rnd_engine); };
//
//	for (int i = 0; i < B.Rows(); i++)
//		for (int j = 0; j < B.Cols(); j++)
//			B(i, j) = rnd();
//	B += Transposed(B) + size * Matrix<float>::Id(size); // makes it positive-definite
//
//	for (int i = 0; i < B.Rows(); i++)
//	{
//		for (int j = 0; j < B.Cols(); j++)
//			if ((i + j) % 2 == 0)
//				A.push_back(j, B(i, j));
//		A.end_row();
//	}
//
//	for (int i = 0; i < x.size(); i++)
//		x[i] = i;
//
//	Vector b = A * x;
//
//	LSQR solver(A, b, _0);
//	solver.Iterate(size);
//
//	EXPECT_LT(solver.Residuum(), 1e-5);
//	EXPECT_LT(norm(solver.Error()), 1e-6);
//}
