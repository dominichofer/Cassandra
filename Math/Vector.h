#pragma once
#include <vector>
#include <string>

using Vector = std::vector<float>;

void operator+=(Vector&, const Vector&);
void operator-=(Vector&, const Vector&);
void operator*=(Vector&, float);
void operator/=(Vector&, float);

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
