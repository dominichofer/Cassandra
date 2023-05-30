#include "PatternFit.h"
#include "Math/Math.h"
#include <iostream>

ScoreEstimator CreateScoreEstimator(
    const std::vector<BitBoard>& pattern,
    const std::vector<Position>& pos,
    const std::vector<float>& score,
    int iterations)
{
    if (pos.size() != score.size())
        throw std::runtime_error("Size mismatch!");

    auto indexer = GroupIndexer(pattern);
    std::size_t elements_per_row = indexer.Variations().size();
    std::size_t rows = pos.size();
    std::size_t cols = indexer.index_space_size;

    MatrixCSR<int> matrix{ elements_per_row, cols, rows };
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < static_cast<int64_t>(rows); i++)
        indexer.InsertIndices(pos[i], matrix.Row(i));

    std::vector<float> weights(cols, 0.0f);

    DiagonalPreconditioner P(matrix.JacobiPreconditionerSquare(100.0f));

    PCG solver(transposed(matrix) * matrix, P, weights, transposed(matrix) * score);
    solver.Iterate(iterations);

    return ScoreEstimator(pattern, solver.X());
}

std::vector<Position> EmptyCountFilter(
    const std::vector<Position>& pos,
    int min_empty_count,
    int max_empty_count)
{
    std::vector<Position> ret;
    for (Position p : pos)
        if (min_empty_count <= p.EmptyCount() and p.EmptyCount() <= max_empty_count)
            ret.push_back(p);
    return ret;
}

MSSE CreateMultiStageScoreEstimator(
    int stage_size,
    const std::vector<BitBoard>& pattern,
    const std::vector<Position>& pos,
    Intensity eval_intensity)
{
    AAMSSE model{ stage_size, pattern };
    HT tt{ 10'000'000 };
    IDAB<PVS> alg{ tt, model };

    for (int stage = 0; stage < model.score_estimator.Stages(); stage++)
    {
        alg.clear();

        int min = stage * stage_size;
        int max = (stage + 1) * stage_size - 1;

        std::vector<Position> stage_train_pos = EmptyCountFilter(pos, min, max);

        auto start = std::chrono::high_resolution_clock::now();
        std::vector<float> train_score(stage_train_pos.size());
        #pragma omp parallel for
        for (int i = 0; i < static_cast<int>(stage_train_pos.size()); i++)
            train_score[i] = alg.Eval(stage_train_pos[i], eval_intensity).score;
        auto stop = std::chrono::high_resolution_clock::now();
        std::cout << "Eval: " << stage_train_pos.size() << " : " << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start) << std::endl;

        start = std::chrono::high_resolution_clock::now();
        auto new_e = CreateScoreEstimator(pattern, stage_train_pos, train_score);
        stop = std::chrono::high_resolution_clock::now();
        std::cout << "Fitting: " << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start) << std::endl;
        model.score_estimator.Weights(stage, new_e.Weights());
    }

    return model.score_estimator;
}

std::pair<AM, double> CreateAccuracyModel(std::span<const PositionMultiDepthScore> data)
{
    struct XY {
        int D, d, e, score_diff;

        auto operator<=>(const XY&) const noexcept = default;

        std::vector<int> x() const { return { D, d, e }; }
        int y() const { return score_diff; }
    };

    std::vector<XY> xy;
    for (const PositionMultiDepthScore& datum : data)
        for (int D = 0; D < datum.score_of_depth.size(); D++)
            if (datum.score_of_depth[D] != undefined_score)
                for (int d = 0; d < D; d++)
                    if (datum.score_of_depth[d] != undefined_score)
                        xy.emplace_back(D, d, datum.pos.EmptyCount(), datum.score_of_depth[D] - datum.score_of_depth[d]);

    std::sort(std::execution::par, xy.begin(), xy.end());
    auto chunks = xy | ranges::views::chunk_by([](const XY& l, const XY& r) { return l.x() == r.x(); });
    auto x = chunks | ranges::views::transform([](auto&& rng) { return ranges::front(rng).x(); }) | ranges::to_vector;
    auto y = chunks | ranges::views::transform([](auto&& rng) { return StandardDeviation(rng | ranges::views::transform(&XY::y)); }) | ranges::to_vector;
    std::size_t x_size = std::ranges::size(x);
    std::size_t y_size = std::ranges::size(y);
    if (x_size != y_size)
        throw std::runtime_error("Size mismatch!");

    AM blank;
    auto param_values = NonLinearLeastSquaresFit(blank.Function(), blank.Parameters(), blank.Variables(), x, y, blank.ParameterValues(), /*steps*/ 1'000, /*damping_factor*/ 0.1);
    AM model(param_values);

    std::vector<double> error(x_size);
    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(x_size); i++)
        error[i] = model.Eval(x[i]) - y[i];

    float R_sq = 1.0f - Variance(error) / Variance(y);
    return std::make_pair(model, R_sq);
}

std::pair<AAMSSE, double> CreateAAMSSE(
    int stage_size,
    const std::vector<BitBoard>& pattern,
    const std::vector<Position>& train_pos,
    const std::vector<Position>& accuracy_pos,
    Intensity eval_intensity,
    int accuracy_max_depth)
{
    MSSE msse = CreateMultiStageScoreEstimator(stage_size, pattern, train_pos, eval_intensity);
    if (accuracy_pos.empty())
        return std::make_pair(AAMSSE{ msse, {} }, 0);

    AAMSSE model{ msse, AM{} };
    HT tt{ 10'000'000 };
    IDAB<PVS> alg{ tt, model };

    std::vector<PositionMultiDepthScore> accuracy_data(accuracy_pos.begin(), accuracy_pos.end());
    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(accuracy_data.size()); i++)
    {
        int e = accuracy_data[i].pos.EmptyCount();
        for (int d = 0; d <= e and d <= accuracy_max_depth; d++)
            accuracy_data[i].score_of_depth[d] = alg.Eval(accuracy_data[i].pos, d).score;
    }

    auto [am, R_sq] = CreateAccuracyModel(accuracy_data);
    return std::make_pair(AAMSSE{ msse, am }, R_sq);
}
