#include "PatternFit.h"
#include "Core/Core.h"
#include "Math/Math.h"
#include "Pattern/Pattern.h"
#include "Search/Search.h"
#include <algorithm>
#include <execution>
#include <map>
#include <stdexcept>

Vector FitWeights(
    const std::vector<uint64_t>& pattern,
    const std::vector<Position>& pos,
    const Vector& score,
    int iterations)
{
    if (pos.size() != score.size())
        throw std::runtime_error("Size mismatch!");

    auto indexer = GroupIndexer(pattern);
    std::size_t elements_per_row = indexer.Variations().size();
    std::size_t rows = pos.size();
    std::size_t cols = indexer.index_space_size;

    MatrixCSR matrix{ elements_per_row, rows, cols };
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < static_cast<int64_t>(rows); i++)
        indexer.InsertIndices(pos[i], matrix.Row(i));

    Vector weights(cols, 0.0f);
    DiagonalPreconditioner P(JacobiPreconditionerOfATA(matrix));
    PCG solver(transposed(matrix) * matrix, P, weights, transposed(matrix) * score);
    solver.Iterate(iterations);
    return solver.X();
}

void ImproveScoreEstimator(
    PatternBasedEstimator& estimator,
    const std::vector<Position>& pos,
    int depth, float confidence_level,
    int fitting_iterations)
{
    LoggingTimer timer;
    const auto pattern = estimator.Pattern();

    HT tt{ 10'000'000 };
    PVS alg{ tt, estimator };

    for (int stage = 0; stage < estimator.score.estimators.size(); stage++)
    {
        tt.clear();

        int stage_size = estimator.score.StageSize();
        int min = stage * stage_size;
        int max = (stage + 1) * stage_size - 1;
        std::vector<Position> stage_pos = EmptyCountFiltered(pos, min, max);

        timer.Start(std::format("Stage {}, evaluated {} pos e{} to e{}", stage, stage_pos.size(), min, max));
        Vector score(stage_pos.size(), 0);
        #pragma omp parallel for
        for (int i = 0; i < static_cast<int>(stage_pos.size()); i++)
            score[i] = alg.Eval(stage_pos[i], { -inf_score, +inf_score }, depth, confidence_level).score;
        timer.Stop();

        timer.Start(std::format("Stage {}, fitting", stage));
        Vector weights = FitWeights(pattern, stage_pos, score, fitting_iterations);
        timer.Stop();

        estimator.score.estimators[stage] = ScoreEstimator(pattern, weights);
    }
}
