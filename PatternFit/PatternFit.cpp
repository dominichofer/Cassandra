#include "PatternFit.h"
#include "Base/Base.h"
#include "Search/Search.h"
#include <algorithm>
#include <format>

Vector FitWeights(
    const std::vector<uint64_t>& pattern,
    std::span<ScoredPosition> scored_pos,
    int iterations)
{
    auto indexer = GroupIndexer(pattern);
    std::size_t elements_per_row = indexer.Variations().size();
    std::size_t rows = scored_pos.size();
    std::size_t cols = indexer.index_space_size;

    // Create matrix
    MatrixCSR matrix{ elements_per_row, rows, cols };
    #pragma omp parallel for
    for (int64_t i = 0; i < static_cast<int64_t>(rows); i++)
        indexer.InsertIndices(scored_pos[i].pos, matrix.Row(i));

    // Create vector
	Vector score(rows);
    #pragma omp parallel for
	for (int64_t i = 0; i < static_cast<int64_t>(rows); i++)
		score[i] = scored_pos[i].score;

    Vector weights(cols, 0.0f);
    DiagonalPreconditioner P(JacobiPreconditionerOfATA(matrix));
    PCG solver(transposed(matrix) * matrix, P, weights, transposed(matrix) * score);
    solver.Iterate(iterations);
    //for (int i = 0; i < iterations; i++)
    //{
    //    solver.Iterate();
    //    std::cout << std::format("Iteration {}, error: {}", i + 1, solver.Residuum()) << std::endl;
    //}
    return solver.X();
}

void EvaluateIteratively(
    PatternBasedEstimator& estimator,
    std::vector<ScoredPosition>& scored_pos,
    Intensity intensity,
    int fitting_iterations,
    bool reevaluate)
{
    LoggingTimer timer;
    RAM_HashTable tt{ 10'000'000 };
    PVS alg{ tt, estimator };
    auto pattern = estimator.Pattern();

    for (int stage = 0; stage < estimator.score.estimators.size(); stage++)
    {
        alg.Clear();

        int stage_size = estimator.StageSize();
        int min = stage * stage_size;
        int max = min + stage_size - 1;

        //timer.Start();
        #pragma omp parallel for schedule(static, 1)
        for (int i = 0; i < static_cast<int>(scored_pos.size()); i++)
        {
			ScoredPosition& sp = scored_pos[i];
			int empty_count = sp.EmptyCount();
            if (min <= empty_count and empty_count <= max)
			    if (reevaluate or not sp.HasScore())
                    sp.score = alg.Eval(sp.pos, intensity).window.lower;
        }
        //timer.Stop(std::format("Stage {}, evaluated e{} to e{}.", stage, min, max));

        //timer.Start();
        std::vector<ScoredPosition> stage_pos;
        for (ScoredPosition& sp : scored_pos)
        {
            int empty_count = sp.EmptyCount();
            if (min <= empty_count and empty_count <= max)
                stage_pos.push_back(sp);
        }
        Vector weights = FitWeights(pattern, stage_pos, fitting_iterations);
        estimator.score.estimators[stage] = ScoreEstimator(pattern, weights);
        //timer.Stop(std::format("Stage {}, fitted.", stage));
    }
}
