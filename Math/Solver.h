#pragma once
#include <memory>
#include <functional>
#include "Algorithm.h"
#include "Vector.h"
#include "MatrixInterface.h"

class IterativeSolver
{
public:
	virtual ~IterativeSolver() = default;

	virtual void Reinitialize() = 0;
	virtual void Reinitialize(Vector x0) = 0;
	virtual void Iterate(int n) = 0;
	virtual double Residuum() const = 0;
	virtual Vector Error() const = 0;
	virtual const Vector& GetX() const = 0;
};

class Preconditioner
{
public:
	virtual Vector apply(const Vector&) const = 0;
	virtual Vector revert(const Vector&) const = 0;
};

class DiagonalPreconditioner : public Preconditioner
{
	Vector d;
public:
	DiagonalPreconditioner(Vector d) : d(std::move(d)) {}

	Vector apply(const Vector& x) const override { return d.elementwise_multiplication(x); }
	Vector revert(const Vector& x) const override { return d.elementwise_division(x); }
};


inline std::tuple<double, Vector> decompose(const Vector& x)
{
	const auto n = norm(x);
	return std::tuple(n, x / n);
}

inline Vector operator*(const std::unique_ptr<IMatrix>& m, const Vector& v) { return m->operator*(v); }

// Conjugate Gradient method
// Solves A * x = b for x.
class CG : public IterativeSolver
{
	std::unique_ptr<IMatrix> A;
	Vector x, b, p, r;
public:
	// A: has to be symmetric and positive-definite.
	template <typename Matrix>
	CG(Matrix A_, Vector x0, Vector b_)
		: A(std::make_unique<Matrix>(std::move(A_))), x(std::move(x0)), b(std::move(b_)), r(A->Cols()), p(A->Cols())
	{
		if (A->Cols() != A->Rows()) throw std::runtime_error("Size mismatch.");
		if (A->Cols() != x.size()) throw std::runtime_error("Size mismatch.");
		if (A->Rows() != b.size()) throw std::runtime_error("Size mismatch.");
		Reinitialize();
	}
	~CG() override = default;

	void Reinitialize() override { p = r = Error(); }

	void Reinitialize(Vector x0) override
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
			const auto r_dot_r_old = dot(r, r);
			const auto A_p = A * p;
			const auto alpha = r_dot_r_old / dot(p, A_p);
			x += alpha * p;
			r -= alpha * A_p;
			const auto beta = dot(r, r) / r_dot_r_old;
			p = r + beta * p;
		}
	}

	double Residuum() const override { return dot(r, r); }
	Vector Error() const override { return b - A * x; }
	const Vector& GetX() const override { return x; }
};

// Preconditioned Conjugate Gradient method
// Solves A * P(y) = b, where P(y) = x, for x.
class PCG : public IterativeSolver
{
	std::unique_ptr<IMatrix> A;
	const Preconditioner& P;
	Vector b, x, z, r, p;
public:
	// A: has to be symmetric and positive-definite.
	template <typename Matrix>
	PCG(Matrix A_, const Preconditioner& P, Vector x0, Vector b_)
		: A(std::make_unique<Matrix>(std::move(A_))), P(P), x(std::move(x0)), b(std::move(b_)), r(A->Cols()), p(A->Cols())
	{
		if (A->Cols() != A->Rows()) throw std::runtime_error("Size mismatch.");
		if (A->Cols() != x.size()) throw std::runtime_error("Size mismatch.");
		if (A->Rows() != b.size()) throw std::runtime_error("Size mismatch.");
		Reinitialize();
	}
	~PCG() override = default;

	void Reinitialize() override
	{
		r = Error();
		p = z = P.apply(r);
	}

	void Reinitialize(Vector x0) override
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
			const auto r_dot_z_old = dot(r, z);
			const auto A_p = A * p;
			const auto alpha = r_dot_z_old / dot(p, A_p);
			x += alpha * p;
			r -= alpha * A_p;
			z = P.apply(r);
			const auto beta = dot(r, z) / r_dot_z_old;
			p = z + beta * p;
		}
	}

	double Residuum() const override { return dot(r, r); }
	Vector Error() const override { return b - A * x; }
	const Vector& GetX() const override { return x; }
};

// Preconditioned Conjugate Gradient method
// Solves At * A * P(y) = b, where P(y) = x, for x.
class PCGLS : public IterativeSolver
{
	std::unique_ptr<IMatrix> A;
	const Preconditioner& P;
	Vector x, b, z, r, p;
public:
	// A: has to be symmetric and positive-definite.
	template <typename Matrix>
	PCGLS(Matrix A_, const Preconditioner& P, Vector x0, Vector b_)
		: A(std::make_unique<Matrix>(std::move(A_))), P(P), x(std::move(x0)), b(std::move(b_)), r(A->Cols()), p(A->Cols())
	{
		if (A->Cols() != A->Rows())
			throw std::runtime_error("Size mismatch.");
		if (A->Cols() != x.size())
			throw std::runtime_error("Size mismatch.");
		if (A->Rows() != b.size())
			throw std::runtime_error("Size mismatch.");

		Reinitialize();
	}
	~PCGLS() override = default;

	void Reinitialize() override
	{
		r = Error();
		p = z = P.apply(r);
	}

	void Reinitialize(Vector x0) override
	{
		x = std::move(x0);
		Reinitialize();
	}

	void Iterate(int n = 1) override
	{
		for (int k = 0; k < n; k++)
		{
			const auto r_dot_z_old = dot(r, z);
			const auto A_p = A * p;
			const auto alpha = r_dot_z_old / dot(p, A_p);
			x += alpha * p;
			r -= alpha * A_p;
			z = P.apply(r);
			const auto beta = dot(r, z) / r_dot_z_old;
			p = z + beta * p;
		}
	}

	double Residuum() const override { return dot(r, r); }
	Vector Error() const override { return b - A * x; }
	const Vector& GetX() const override { return x; }
};

// Least Squares QR Method
// Solves A * x = b for x, or minimizes ||A*x-b||.
template <typename Matrix>
class LSQR : public IterativeSolver
{
	// Source of algorithm: https://web.stanford.edu/group/SOL/software/lsqr/lsqr-toms82a.pdf
	const Matrix& A;
	const Vector& b;
	Vector x, u, v, w;
	double alpha{ 0 }, beta{ 0 }, phi{ 0 }, rho{ 0 }, phi_bar{ 0 }, rho_bar{ 0 }, residuum;
public:
	// A: has to be symmetric and positive-definite.
	LSQR(const Matrix& A, Vector x0, const Vector& b)
		: A(A), b(b), x(std::move(x0)), u(A.Rows()), v(A.Cols()), w(A.Cols())
	{
		if (A.Cols() != x.size()) throw;
		if (A.Rows() != b.size()) throw;
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

	void Reinitialize(Vector x0) override
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
	Vector Error() const override { return b - A * x; }
	const Vector& GetX() const override { return x; }
};
