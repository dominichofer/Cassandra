#include "pch.h"
#include <random>

TEST(Mat_Vec, Multiplication)
{
	Matrix<float> A(4, 3);
	Vector x(3);

	for (int i = 0; i < A.Rows(); i++)
		for (int j = 0; j < A.Cols(); j++)
			A(i, j) = i * 10 + j;

	for (int i = 0; i < x.size(); i++)
		x[i] = i;

	Vector b(4);
	for (int i = 0; i < A.Rows(); i++)
	{
		b[i] = 0;
		for (int j = 0; j < A.Cols(); j++)
			b[i] += A(i, j) * x[j];
	}

	EXPECT_TRUE(A * x == b);
}

//TEST(MatCSR_Vec, Multiplication)
//{
//	MatrixCSR<float> A(3);
//	Vector x(3);
//
//	for (int i = 0; i < 4; i++)
//	{
//		for (int j = 0; j < A.Cols(); j++)
//			if ((i + j) % 2 == 0)
//				A.push_back(j, i * 10 + j);
//		A.end_row();
//	}
//
//	for (int i = 0; i < x.size(); i++)
//		x[i] = i;
//
//	Vector b(4);
//	for (int i = 0; i < A.Rows(); i++)
//	{
//		b[i] = 0;
//		for (int j = 0; j < A.Cols(); j++)
//			if ((i + j) % 2 == 0)
//				b[i] += (i * 10 + j) * x[j];
//	}
//
//	auto A_x = A * x;
//	EXPECT_EQ(A_x, b);
//}

TEST(CG, Mat_Vec)
{
	const std::size_t size = 4;
	Matrix<float> A(size, size);
	Vector x(size);
	Vector x0(size);

	std::mt19937_64 rnd_engine(13);
	auto rnd = [&rnd_engine]() { return std::uniform_real_distribution<float>(0, 1)(rnd_engine); };

	for (int i = 0; i < A.Rows(); i++)
		for (int j = 0; j < A.Cols(); j++)
			A(i, j) = rnd();
	A += Transposed(A) + size * Matrix<float>::Id(size); // makes it positive-definite

	for (int i = 0; i < x.size(); i++)
		x[i] = i;

	Vector b = A * x;

	CG cg(A, b, x0);
	cg.Iterate(size);

	EXPECT_LT(cg.Residuum(), 1e-8);
	EXPECT_LT(norm(cg.Error()), 1e-6);
}

//TEST(CG, MatCSR_Vec)
//{
//	const std::size_t size = 4;
//	Matrix<float> B(size, size);
//	MatrixCSR<float, uint8_t> A(size);
//	Vector x(size);
//	Vector x0(size);
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
//	CG solver(A, b, x0);
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
//	Vector x0(size);
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
//	LSQR solver(A, b, x0);
//	solver.Iterate(size);
//
//	EXPECT_LT(solver.Residuum(), 1e-5);
//	EXPECT_LT(norm(solver.Error()), 1e-6);
//}
