#include "pch.h"
#include "Vector.h"

Vector CreateVector(double a)
{
    return Vector(std::vector<double>{a});
}

Vector CreateVector(double a, double b)
{
    return Vector(std::vector<double>{a, b});
}

TEST(VectorTest, Constructor)
{
    Vector v(3, 1.0);
    EXPECT_EQ(v.size(), 3);
    EXPECT_EQ(v[0], 1.0);
    EXPECT_EQ(v[1], 1.0);
    EXPECT_EQ(v[2], 1.0);
}

TEST(VectorTest, Addition)
{
    Vector v1(3, 1.0);
    Vector v2(3, 2.0);
    Vector sum = v1 + v2;

    EXPECT_EQ(sum.size(), 3);
    EXPECT_EQ(sum[0], 3.0);
    EXPECT_EQ(sum[1], 3.0);
    EXPECT_EQ(sum[2], 3.0);
}

TEST(VectorTest, Subtraction)
{
    Vector v1(3, 2.0);
    Vector v2(3, 1.0);
    Vector diff = v1 - v2;

    EXPECT_EQ(diff.size(), 3);
    EXPECT_EQ(diff[0], 1.0);
    EXPECT_EQ(diff[1], 1.0);
    EXPECT_EQ(diff[2], 1.0);
}

TEST(VectorTest, ScalarMultiplication)
{
    Vector v(3, 2.0);
    Vector scaled = v * 3.0;

    EXPECT_EQ(scaled.size(), 3);
    EXPECT_EQ(scaled[0], 6.0);
    EXPECT_EQ(scaled[1], 6.0);
    EXPECT_EQ(scaled[2], 6.0);
}

TEST(VectorTest, ScalarDivision)
{
    Vector v(3, 6.0);
    Vector divided = v / 2.0;

    EXPECT_EQ(divided.size(), 3);
    EXPECT_EQ(divided[0], 3.0);
    EXPECT_EQ(divided[1], 3.0);
    EXPECT_EQ(divided[2], 3.0);
}

TEST(VectorTest, ElementwiseMultiplication)
{
    Vector v1(3, 2.0);
    Vector v2(3, 3.0);
    Vector elemMult = v1.elementwise_multiplication(v2);

    EXPECT_EQ(elemMult.size(), 3);
    EXPECT_EQ(elemMult[0], 6.0);
    EXPECT_EQ(elemMult[1], 6.0);
    EXPECT_EQ(elemMult[2], 6.0);
}

TEST(VectorTest, ElementwiseDivision)
{
    Vector v1(3, 6.0);
    Vector v2(3, 2.0);
    Vector elemDiv = v1.elementwise_division(v2);

    EXPECT_EQ(elemDiv.size(), 3);
    EXPECT_EQ(elemDiv[0], 3.0);
    EXPECT_EQ(elemDiv[1], 3.0);
    EXPECT_EQ(elemDiv[2], 3.0);
}

TEST(VectorTest, Inverse)
{
    Vector v(std::vector{1.0, 2.0, 4.0});
    Vector inverse = inv(v);

    EXPECT_EQ(inverse.size(), 3);
    EXPECT_EQ(inverse[0], 1.0);
    EXPECT_EQ(inverse[1], 0.5);
    EXPECT_EQ(inverse[2], 0.25);
}

TEST(VectorTest, DotProduct)
{
    Vector v1(3, 2.0);
    Vector v2(3, 3.0);
    double dotProduct = dot(v1, v2);

    EXPECT_EQ(dotProduct, 18.0);
}

TEST(VectorTest, L1Norm)
{
    Vector v(3, -2.0);
    double l1Norm = L1_norm(v);

    EXPECT_DOUBLE_EQ(l1Norm, 6.0);
}

TEST(VectorTest, L2Norm)
{
    Vector v(std::vector<double>{3.0, 4.0});
    double l2Norm = L2_norm(v);

    EXPECT_DOUBLE_EQ(l2Norm, 5.0);
}

TEST(VectorTest, Sum)
{
    Vector v(3, 2.0);
    double result = sum(v);

    EXPECT_DOUBLE_EQ(result, 6.0);
}

TEST(VectorTest, SizeMismatch)
{
    Vector v1(3, 2.0);
    Vector v2(4, 3.0);

    EXPECT_THROW(v1 += v2, std::runtime_error);
    EXPECT_THROW(v1 -= v2, std::runtime_error);
    EXPECT_THROW(v1.elementwise_multiplication(v2), std::runtime_error);
    EXPECT_THROW(v1.elementwise_division(v2), std::runtime_error);
    EXPECT_THROW(dot(v1, v2), std::runtime_error);
}
