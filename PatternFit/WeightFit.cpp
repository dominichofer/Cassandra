#include "WeightFit.h"
#include "Math/MatrixCSR.h"
#include "Math/Solver.h"
#include "Math/Statistics.h"
#include "Pattern/DenseIndexer.h"

MatrixCSR<uint32_t> CreateMatrix(const DenseIndexer& indexer, const std::vector<Position>& pos)
{
    const int64_t size = static_cast<int64_t>(pos.size());
    const auto row_size = indexer.variations;
    const auto cols = indexer.reduced_size;
    const auto rows = size;
    MatrixCSR<uint32_t> mat(row_size, cols, rows);

    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < size; i++)
        for (int j = 0; j < indexer.variations; j++)
            mat.begin(i)[j] = indexer.DenseIndex(pos[i], j);
    return mat;
}

Pattern::Weights FitWeights(const std::vector<BitBoard>& pattern, const WeightFit::Data& data)
{
    auto indexer = CreateDenseIndexer(pattern);
    auto matrix = CreateMatrix(*indexer, data.Pos());
    Vector weights(indexer->reduced_size, 0);
    DiagonalPreconditioner P(matrix.JacobiPreconditionerSquare(1000));
    PCG solver(transposed(matrix) * matrix, P, weights, transposed(matrix) * data.Score());
    solver.Iterate(10);
    return solver.X();
}

std::vector<float> EvalErrors(const Pattern::Weights& weights, const std::vector<BitBoard>& pattern, const WeightFit::Data& data)
{
    auto indexer = CreateDenseIndexer(pattern);
    auto matrix = CreateMatrix(*indexer, data.Pos());
    return data.Score() - matrix * weights;
}

std::tuple<double, double> EvalAccuracy(const Pattern::Weights& weights, const std::vector<BitBoard>& pattern, const WeightFit::DataPair& data)
{
    auto train_error = EvalErrors(weights, pattern, data.train);
    auto test_error = EvalErrors(weights, pattern, data.test);
    return std::make_tuple(StandardDeviation(train_error), StandardDeviation(test_error));
}