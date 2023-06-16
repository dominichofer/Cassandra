#pragma once
#include <vector>

class Vector
{
	std::vector<double> data;
public:
	Vector(std::size_t count, double value) noexcept : data(count, value) {}
	Vector(const std::vector<double>& o) noexcept : data(o) {}
	Vector(std::vector<double>&& o) noexcept : data(std::move(o)) {}

	Vector(const Vector& o) noexcept : data(o.data) {}
	Vector(Vector&& o) noexcept : data(std::move(o.data)) {}

	Vector operator=(const Vector& o) noexcept { data = o.data; }
	Vector operator=(Vector&& o) noexcept { data = std::move(o.data); }

	bool operator==(const Vector& o) const noexcept { return data == o.data; }
	bool operator!=(const Vector& o) const noexcept { return !(*this == o); }

	      double& operator[](std::size_t index)       { return data[index]; }
	const double& operator[](std::size_t index) const { return data[index]; }

	std::size_t size() const noexcept { return data.size(); }

	auto begin() noexcept { return data.begin(); }
	auto begin() const noexcept { return data.begin(); }
	auto end() noexcept { return data.end(); }
	auto end() const noexcept { return data.end(); }

	Vector& operator+=(const Vector&);
	Vector& operator-=(const Vector&);
	Vector& operator*=(double);
	Vector& operator/=(double);
	Vector& elementwise_multiplication(const Vector&);
	Vector& elementwise_division(const Vector&);
};

Vector operator+(Vector, const Vector&);
Vector operator+(const Vector&, Vector&&);

Vector operator-(Vector, const Vector&);
Vector operator-(const Vector&, Vector&&);

Vector operator*(Vector, double);
Vector operator*(double, Vector);

Vector operator/(Vector, double);

Vector inv(Vector);
double dot(const Vector&, const Vector&);
double norm(const Vector&);
double L1_norm(const Vector&);
double L2_norm(const Vector&);
double sum(const Vector&);
