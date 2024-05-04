#pragma once
#include "Vector.h"
#include "Matrix.h"
#include <memory>
#include <functional>
#include <stdexcept>
#include <iostream>

class IterativeSolver
{
public:
	virtual void Iterate(int n) = 0;
	virtual float Residuum() const = 0;
	virtual Vector Error() const = 0;
	virtual Vector X() const = 0;
};

class Preconditioner
{
public:
	virtual Vector apply(const Vector&) const = 0;
	virtual Vector revert(const Vector&) const = 0;
};

class DiagonalPreconditioner : public Preconditioner
{
public:
	Vector d; // TODO: Make this private!
	DiagonalPreconditioner(Vector d) : d(std::move(d))
	{
		for (float& x : this->d)
			if (not std::isnormal(x))
				x = 1;
	}

	Vector apply(const Vector& x) const override { return elementwise_multiplication(d, x); }
	Vector revert(const Vector& x) const override { return elementwise_division(d, x); }
};

// Conjugate Gradient method
// Solves A * x = b for x.
template <typename Matrix>
class CG final : public IterativeSolver
{
	const Matrix& A;
	Vector x, b, p, r;
public:
	// A: has to be symmetric and positive-definite.
	CG(const Matrix& A, Vector x0, Vector b)
		: A(A), x(std::move(x0)), b(std::move(b))
	{
		if (A.Cols() != A.Rows()) throw std::runtime_error("Size mismatch.");
		if (A.Cols() != x.size()) throw std::runtime_error("Size mismatch.");
		if (A.Rows() != this->b.size()) throw std::runtime_error("Size mismatch.");

		p = r = Error();
	}

	void Iterate(int n = 1) override
	{
		for (int k = 0; k < n; k++)
		{
			float r_dot_r_old = dot(r, r);
			Vector A_p = A * p;
			float alpha = r_dot_r_old / dot(p, A_p);
			x += alpha * p;
			r -= alpha * A_p;
			float beta = dot(r, r) / r_dot_r_old;
			p = r + beta * p;
		}
	}

	float Residuum() const override { return dot(r, r); }
	Vector Error() const override { return b - A * x; }
	Vector X() const override { return x; }
};

// Preconditioned Conjugate Gradient method
// Solves A * P(y) = b for x, where P(y) = x.
template <typename Matrix>
class PCG final : public IterativeSolver
{
	const Matrix& A;
	const Preconditioner& P;
	Vector b, x, z, r, p;
public:
	// A: has to be symmetric and positive-definite.
	PCG(const Matrix& A, const Preconditioner& P, Vector x0, Vector b)
		: A(A), P(P), x(std::move(x0)), b(std::move(b))
	{
		if (A.Rows() != A.Cols()) throw std::runtime_error("Size mismatch.");
		if (A.Cols() != x.size()) throw std::runtime_error("Size mismatch.");
		if (A.Rows() != this->b.size()) throw std::runtime_error("Size mismatch.");

		r = Error();
		p = z = P.apply(r);
	}

	void Iterate(int n = 1) override
	{
		for (int k = 0; k < n; k++)
		{
			float r_dot_z_old = dot(r, z);
			Vector A_p = A * p;
			float alpha = r_dot_z_old / dot(p, A_p);
			x += alpha * p;
			r -= alpha * A_p;
			z = P.apply(r);
			float beta = dot(r, z) / r_dot_z_old;
			p = z + beta * p;
		}
	}

	float Residuum() const override { return dot(r, r); }
	Vector Error() const override { return b - A * x; }
	Vector X() const override { return x; }
};

// Decomposes vector into a unit vector and its length.
inline std::pair<float, Vector> decompose(const Vector& v)
{
	auto length = norm(v);
	if (length == 0)
		return std::make_pair(0.0f, Vector(v.size(), 0));
	return std::make_pair(length, v / length);
}

// Least Squares QR Method
// Solves A * x = b for x, or minimizes ||A*x-b||.
template <typename Matrix>
class LSQR : public IterativeSolver
{
	// Source of algorithm: https://web.stanford.edu/group/SOL/software/lsqr/lsqr-toms82a.pdf
	const Matrix& A;
	const Vector& b;
	Vector x, u, v, w;
	float alpha, phi_bar, rho_bar;
public:
	LSQR(const Matrix& A, Vector x0, Vector b)
		: A(A), x(std::move(x0)), b(std::move(b))
	{
		if (A.Cols() != x.size()) throw std::runtime_error("Size mismatch.");
		if (A.Rows() != this->b.size()) throw std::runtime_error("Size mismatch.");

		float beta;
		std::tie(beta, u) = decompose(this->b);
		std::tie(alpha, v) = decompose(transposed(A) * u);
		w = v;
		phi_bar = beta;
		rho_bar = alpha;
	}

	void Iterate(int n = 1) override
	{
		// 2n x O(mat*vec)
		// 4n x O(vec)
		for (int k = 0; k < n; k++)
		{
			// Continue the bidiagonalization
			float beta;
			std::tie(beta, u) = decompose(A * v - alpha * u);
			std::tie(alpha, v) = decompose(transposed(A) * u - beta * v);

			// Construct and apply next orthogonal transformation
			float rho = std::sqrt(rho_bar * rho_bar + beta * beta);
			float c = rho_bar / rho;
			float s = beta / rho;
			float theta = s * alpha;
			rho_bar = -c * alpha;
			float phi = c * phi_bar;
			phi_bar = s * phi_bar;

			// Update x, w
			x += (phi / rho) * w;
			w = v - (theta / rho) * w;
		}
	}

	float Residuum() const override { return phi_bar; }
	Vector Error() const override { return b - A * x; }
	Vector X() const override { return x; }
};

inline float signum(float value)
{
	if (value < 0)
		return -1.0f;
	if (value > 0)
		return +1.0f;
	return 0.0f;
}

inline std::tuple<float, float, float> GivensRotation(float a, float b)
{
	// From: https://github.com/scipy/scipy/blob/v1.10.1/scipy/sparse/linalg/_isolve/lsqr.py

	if (b == 0)
		return std::make_tuple(signum(a), 0.0f, std::abs(a));
	if (a == 0)
		return std::make_tuple(0.0f, signum(b), std::abs(b));
	if (std::abs(b) > std::abs(a))
	{
		float tau = a / b;
		float s = signum(b) / std::sqrt(1 + tau * tau);
		float c = s * tau;
		float r = b / s;
		return std::make_tuple(c, s, r);
	}
	else
	{
		float tau = b / a;
		float c = signum(a) / std::sqrt(1 + tau * tau);
		float s = c * tau;
		float r = a / c;
		return std::make_tuple(c, s, r);
	}
}

// Least Squares Minres Method
// Solves A * x = b for x, or minimizes ||A*x-b||.
template <typename Matrix>
class LSMR : public IterativeSolver
{
	// Source of algorithm: https://web.stanford.edu/group/SOL/software/lsqr/lsqr-toms82a.pdf
	const Matrix& A;
	const Vector& b;
	Vector x, v, u, h, h_bar;
	float alpha, alpha_bar, zeta_bar, c_bar, s_bar, rho, rho_bar;
public:
	LSMR(const Matrix& A, Vector x0, Vector b)
		: A(A), x(std::move(x0)), b(std::move(b))
	{
		if (A.Cols() != x.size()) throw std::runtime_error("Size mismatch.");
		if (A.Rows() != this->b.size()) throw std::runtime_error("Size mismatch.");

		float beta;
		std::tie(beta, u) = decompose(this->b);
		std::tie(alpha, v) = decompose(transposed(A) * u);
		alpha_bar = alpha;
		zeta_bar = alpha * beta;
		rho = 1;
		rho_bar = 1;
		c_bar = 1;
		s_bar = 0;
		h = v;
		h_bar = Vector(h.size(), 0);
	}

	void Iterate(int n = 1) override
	{
		for (int k = 0; k < n; k++)
		{
			float rho_old = rho;
			float rho_bar_old = rho_bar;

			// Continue the bidiagonalization
			float beta;
			std::tie(beta, u) = decompose(A * v - alpha * u);
			std::tie(alpha, v) = decompose(transposed(A) * u - beta * v);

			// Construct rotation Qhat_{k,2k+1}
			auto [c_hat, s_hat, alpha_hat] = GivensRotation(alpha_bar, /*damp*/ 0);

			// Use a plane rotation(Q_i) to turn B_i to R_i
			float c, s;
			std::tie(c, s, rho) = GivensRotation(alpha_hat, beta);
			float theta_new = s * alpha;
			alpha_bar = c * alpha;

			// Use a plane rotation(Qbar_i) to turn R_i ^ T to R_i ^ bar
			float theta_bar = s_bar * rho;
			float rho_temp = c_bar * rho;
			std::tie(c_bar, s_bar, rho_bar) = GivensRotation(c_bar * rho, theta_new);
			float zeta = c_bar * zeta_bar;
			zeta_bar *= -s_bar;

			// Update h, h_bar, x
			h_bar = h - (theta_bar * rho / rho_old / rho_bar_old) * h_bar;
			x += zeta / rho / rho_bar * h_bar;
			h = v - theta_new / rho * h;
		}
	}

	float Residuum() const override { return 0; }
	Vector Error() const override { return b - A * x; }
	Vector X() const override { return x; }
};
