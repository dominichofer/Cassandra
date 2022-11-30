#include "pch.h"
#include <random>


TEST(MatrixCSR, mat_vec_mult)
{
	MatrixCSR<int> A(1, 3, 4);
	DenseMatrix<int> AA(4, 3);
	*(A.begin() + 0) = 0; AA(0,0) = 1;
	*(A.begin() + 1) = 2; AA(1,2) = 1;
	*(A.begin() + 2) = 1; AA(2,1) = 1;
	*(A.begin() + 3) = 0; AA(3,0) = 1;

	std::vector x = { 1,2,3 };
	EXPECT_TRUE(A * x == AA * x);

	x = { 1,2,3,4 };
	EXPECT_TRUE(transposed(A) * x == transposed(AA) * x);
	EXPECT_TRUE(x * A == x * AA);
}

TEST(CG, DenseMatrix)
{
	const std::size_t size = 4;
	DenseMatrix<float> A(size, size);
	std::vector<float> x = {1,2,3,4};
	std::vector<float> x0(size);

	std::mt19937_64 rnd_engine(13);
	auto rnd = [&rnd_engine]() { return std::uniform_real_distribution<float>(0, 1)(rnd_engine); };

	for (int i = 0; i < A.Rows(); i++)
		for (int j = 0; j < A.Cols(); j++)
			A(i, j) = rnd();
	A += transposed(A) + static_cast<float>(size) * DenseMatrix<float>::Id(size); // makes it positive definite

	std::vector b = A * x;

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
//	std::vector x(size);
//	std::vector _0(size);
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
//	std::vector b = A * x;
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
//	std::vector x(size);
//	std::vector _0(size);
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
//	std::vector b = A * x;
//
//	LSQR solver(A, b, _0);
//	solver.Iterate(size);
//
//	EXPECT_LT(solver.Residuum(), 1e-5);
//	EXPECT_LT(norm(solver.Error()), 1e-6);
//}
