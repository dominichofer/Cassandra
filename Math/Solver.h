#pragma once
#include "Algorithm.h"
#include "Vector.h"
#include "Matrix.h"
#include <memory>
#include <functional>
#include <stdexcept>

template <typename T>
class IterativeSolver
{
public:
	virtual ~IterativeSolver() = default;

	virtual void Reinitialize() = 0;
	virtual void Reinitialize(std::vector<T>) = 0;
	virtual void Iterate(int n) = 0;
	virtual double Residuum() const = 0;
	virtual std::vector<T> Error() const = 0;
	virtual std::vector<T> X() const = 0;
};

template <typename T>
class Preconditioner
{
public:
	virtual std::vector<T> apply(const std::vector<T>&) const = 0;
	virtual std::vector<T> revert(const std::vector<T>&) const = 0;
};

template <typename T>
class DiagonalPreconditioner : public Preconditioner<T>
{
	std::vector<T> d;
public:
	DiagonalPreconditioner(std::vector<T> d) : d(std::move(d)) {}

	std::vector<T> apply(const std::vector<T>& x) const override { return elementwise_multiplication(d, x); }
	std::vector<T> revert(const std::vector<T>& x) const override { return elementwise_division(d, x); }
};

// Conjugate Gradient method
// Solves A * x = b for x.
template <typename T, typename Matrix>
class CG final : public IterativeSolver<T>
{
	const Matrix& A;
	std::vector<T> x, b, p, r;
public:
	// A: has to be symmetric and positive-definite.
	CG(const Matrix& A, std::vector<T> x0, std::vector<T> b)
		: A(A), x(std::move(x0)), b(std::move(b))
	{
		if (A.Cols() != A.Rows()) throw std::runtime_error("Size mismatch.");
		if (A.Cols() != x.size()) throw std::runtime_error("Size mismatch.");
		if (A.Rows() != this->b.size()) throw std::runtime_error("Size mismatch.");
		Reinitialize();
	}
	~CG() override = default;

	void Reinitialize() override { p = r = Error(); }

	void Reinitialize(std::vector<T> start) override
	{
		x = std::move(start);
		Reinitialize();
	}

	void Iterate(int n = 1) override
	{
		// 1n x O(mat*vec)
		// 10n x O(vec)  Room for optimization to 5n x O(vec)
		for (int k = 0; k < n; k++)
		{
			const double r_dot_r_old = dot(r, r);
			const auto A_p = A * p;
			const double alpha = r_dot_r_old / dot(p, A_p);
			x += alpha * p;
			r -= alpha * A_p;
			const double beta = dot(r, r) / r_dot_r_old;
			p = r + beta * p;
		}
	}

	double Residuum() const override { return dot(r, r); }
	std::vector<T> Error() const override { return b - A * x; }
	std::vector<T> X() const override { return x; }
};

// Preconditioned Conjugate Gradient method
// Solves A * P(y) = b, where P(y) = x, for x.
template <typename T, typename Matrix>
class PCG final : public IterativeSolver<T>
{
	const Matrix& A;
	const Preconditioner<T>& P;
	std::vector<T> b, x, z, r, p;
public:
	// A: has to be symmetric and positive-definite.
	PCG(const Matrix& A, const Preconditioner<T>& P, std::vector<T> x0, std::vector<T> b)
		: A(A), P(P), x(std::move(x0)), b(std::move(b))
	{
		if (A.Rows() != A.Cols()) throw std::runtime_error("Size mismatch.");
		if (A.Cols() != x.size()) throw std::runtime_error("Size mismatch.");
		if (A.Rows() != this->b.size()) throw std::runtime_error("Size mismatch.");
		Reinitialize();
	}
	~PCG() override = default;

	void Reinitialize() override
	{
		r = Error();
		p = z = P.apply(r);
	}

	void Reinitialize(std::vector<T> x0) override
	{
		x = std::move(x0);
		Reinitialize();
	}

	void Iterate(int n = 1) override
	{
		// 1n x O(mat*vec)
		// 10n x O(vec)  Room for optimization to 5n x O(vec)
		for (int k = 0; k < n; k++)
		{
			const T r_dot_z_old = dot(r, z);
			const std::vector<T> A_p = A * p;
			const T alpha = r_dot_z_old / dot(p, A_p);
			if (not std::isnormal(alpha))
				break;
			x += alpha * p;
			r -= alpha * A_p;
			z = P.apply(r);
			const double beta = dot(r, z) / r_dot_z_old;
			p = z + beta * p;
		}
	}

	double Residuum() const override { return dot(r, r); }
	std::vector<T> Error() const override { return b - A * x; }
	std::vector<T> X() const override { return x; }
};

// Least Squares QR Method
// Solves A * x = b for x, or minimizes ||A*x-b||.
template <typename T, typename Matrix>
class LSQR : public IterativeSolver<T>
{
	// Source of algorithm: https://web.stanford.edu/group/SOL/software/lsqr/lsqr-toms82a.pdf
	const Matrix& A;
	const std::vector<T>& b;
	std::vector<T> x, u, v, w;
	double alpha{ 0 }, beta{ 0 }, phi{ 0 }, rho{ 0 }, phi_bar{ 0 }, rho_bar{ 0 }, residuum;
public:
	// A: has to be symmetric and positive-definite.
	LSQR(const Matrix& A, std::vector<T> x0, std::vector<T> b)
		: A(A), b(std::move(b)), x(std::move(x0)), u(A.Rows()), v(A.Cols()), w(A.Cols())
	{
		if (A.Cols() != x.size()) throw;
		if (A.Rows() != this->b.size()) throw;
		Reinitialize();
	}
	~LSQR() override = default;

	void Reinitialize() override
	{
		std::tie(beta, u) = decompose(b - A * x);
		std::tie(alpha, v) = decompose(transposed(A) * u);
		w = v;
		phi_bar = beta;
		rho_bar = alpha;
		rho = sqrt(rho_bar * rho_bar + beta * beta);
		residuum = phi_bar * alpha * std::abs(rho_bar / rho);
	}

	void Reinitialize(std::vector<T> x0) override
	{
		x = std::move(x0);
		Reinitialize();
	}

	void Iterate(int n = 1) override
	{
		// 2n x O(mat*vec)
		// 7n x O(vec)  Room for optimization to 5n x O(vec)
		for (int k = 0; k < n; k++)
		{
			std::tie(beta, u) = decompose(A * v - alpha * u);
			std::tie(alpha, v) = decompose(transposed(A) * u - beta * v);
			rho = std::sqrt(rho_bar * rho_bar + beta * beta);
			const double c = rho_bar / rho;
			const double s = beta / rho;
			const double theta = s * alpha;
			rho_bar = -c * alpha;
			phi = c * phi_bar;
			phi_bar = s * phi_bar;
			x += (phi / rho) * w;
			w = v - (theta / rho) * w;
			residuum = phi_bar * alpha * std::abs(c);
		}
	}

	double Residuum() const override { return residuum; }
	std::vector<T> Error() const override { return b - A * x; }
	std::vector<T> X() const override { return x; }
};
