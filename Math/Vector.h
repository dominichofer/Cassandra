#pragma once
#include <vector>
#include <string>

class Vector
{
	std::vector<float> data;
public:
	Vector() noexcept = default;
	Vector(std::size_t count, float value) noexcept : data(count, value) {}
	Vector(const std::vector<float>& o) noexcept : data(o) {}
	Vector(std::vector<float>&& o) noexcept : data(std::move(o)) {}

	Vector(const Vector& o) noexcept : data(o.data) {}
	Vector(Vector&& o) noexcept : data(std::move(o.data)) {}

	operator const std::vector<float>&() const noexcept { return data; }

	Vector& operator=(const Vector& o) noexcept { data = o.data; return *this; }
	Vector& operator=(Vector&& o) noexcept { data = std::move(o.data); return *this; }

	bool operator==(const Vector& o) const noexcept { return data == o.data; }
	bool operator!=(const Vector& o) const noexcept { return !(*this == o); }

	      float& operator[](std::size_t index)       { return data[index]; }
	const float& operator[](std::size_t index) const { return data[index]; }

	std::size_t size() const noexcept { return data.size(); }

	auto begin() noexcept { return data.begin(); }
	auto begin() const noexcept { return data.begin(); }
	auto end() noexcept { return data.end(); }
	auto end() const noexcept { return data.end(); }

	Vector& operator+=(const Vector&);
	Vector& operator-=(const Vector&);
	Vector& operator*=(float);
	Vector& operator/=(float);
	Vector& elementwise_multiplication(const Vector&);
	Vector& elementwise_division(const Vector&);
};

Vector operator+(Vector, const Vector&);
Vector operator+(const Vector&, Vector&&);

Vector operator-(Vector, const Vector&);
Vector operator-(const Vector&, Vector&&);

Vector operator*(Vector, float);
Vector operator*(float, Vector);

Vector operator/(Vector, float);

Vector elementwise_multiplication(Vector, const Vector&);
Vector elementwise_division(Vector, const Vector&);

Vector inv(Vector);
float dot(const Vector&, const Vector&);
float norm(const Vector&);
float L1_norm(const Vector&);
float L2_norm(const Vector&);
float sum(const Vector&);

std::string to_string(const Vector&);
