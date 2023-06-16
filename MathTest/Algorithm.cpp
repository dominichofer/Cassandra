#include "pch.h"
#include "Matrix.h"
#include "Vector.h"
TEST(Algorithm, CholeskyDecomposition)
{
    Matrix A = CreateMatrix2x2(4, 2, 2, 17);
    Matrix L = CholeskyDecomposition(A);

    ASSERT_EQ(L, CreateMatrix2x2(2, 0, 1, 4));
}

TEST(Algorithm, ForwardSubstitution)
{
    // |1 0| * |1| = |1|
    // |2 3|   |2|   |8|

    Matrix L = CreateMatrix2x2(1, 0, 2, 3);
    Vector b = CreateVector(1, 8);

    Vector x = ForwardSubstitution(L, b);

    EXPECT_EQ(x, CreateVector(1, 2));
}

TEST(Algorithm, BackwardSubstitution)
{
    // |1 2| * |1| = |5|
    // |0 3|   |2|   |6|

    Matrix U = CreateMatrix2x2(1, 2, 0, 3);
    Vector b = CreateVector(5, 6);

    Vector x = BackwardSubstitution(U, b);

    EXPECT_EQ(x, CreateVector(1, 2));
}

//TEST(Algorithm, GaussNewtonStep)
//{
//    // Define the function and variables
//    SymExp function = SymExp::Sin(Var{ "x" } *Var{ "a" }) + Var{ "b" };
//    Vars params = { Var{ "a" }, Var{ "b" } };
//    Vars vars = { Var{ "x" } };
//
//    // Define the data points
//    std::vector<double> x = { 0.0, 1.0, 2.0 };
//    std::vector<double> y = { 0.0, 1.0, 0.0 };
//
//    // Initial parameter values
//    Vector param_values = { 1.0, 1.0 };
//
//    double damping_factor = 0.01;
//
//    // Expected result
//    Vector expected = { 0.999983, 0.009908 };
//
//    Vector result = GaussNewtonStep(function, params, vars, x, y, param_values, damping_factor);
//
//    // Check each parameter value within a tolerance
//    for (int i = 0; i < result.size(); ++i)
//    {
//        ASSERT_NEAR(result[i], expected[i], 1e-5);
//    }
//}

//TEST(Algorithm, NonLinearLeastSquaresFit)
//{
//    // Define the function and variables
//    SymExp function = SymExp::Sin(Var{ "x" } *Var{ "a" }) + Var{ "b" };
//    Vars params = { Var{ "a" }, Var{ "b" } };
//    Vars vars = { Var{ "x" } };
//
//    // Define the data points
//    std::vector<double> x = { 0.0, 1.0, 2.0 };
//    std::vector<double> y = { 0.0, 1.0, 0.0 };
//
//    // Initial parameter values
//    Vector param_values = { 1.0, 1.0 };
//
//    int steps = 10;
//    double damping_factor = 0.01;
//
//    // Expected result
//    Vector expected = { 0.999983, 0.009908 };
//
//    Vector result = NonLinearLeastSquaresFit(function, params, vars, x, y, param_values, steps, damping_factor);
//
//    // Check each parameter value within a tolerance
//    for (int i = 0; i < result.size(); ++i)
//    {
//        ASSERT_NEAR(result[i], expected[i], 1e-5);
//    }
//}
