#include "PatternFit.h"
#include <vector>

void Fit(GLEM& model, const std::vector<PosScore>& data, int iterations)
{
    auto indexer = GroupIndexer(model.Pattern());
    std::size_t elements_per_row = indexer.Variations().size();
    std::size_t rows = data.size();
    std::size_t cols = indexer.index_space_size;

    MatrixCSR<int> matrix{ elements_per_row, cols, rows };
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < static_cast<int64_t>(rows); i++)
        indexer.InsertIndices(data[i].pos, matrix.Row(i));

    std::vector<float> score(rows);
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < static_cast<int64_t>(rows); i++)
        score[i] = data[i].score;

    DiagonalPreconditioner P(matrix.JacobiPreconditionerSquare(100.0f));

    PCG solver(transposed(matrix) * matrix, P, model.Weights(), transposed(matrix) * score);
    solver.Iterate(iterations);

    model.SetWeights(solver.X());
}

void Fit(GLEM& model, std::vector<Position>::const_iterator pos_begin, std::vector<Position>::const_iterator pos_end, const std::vector<float>& score, int iterations)
{
    if (std::distance(pos_begin, pos_end) != score.size())
        throw std::runtime_error("Size mismatch");

    auto indexer = GroupIndexer(model.Pattern());
    std::size_t elements_per_row = indexer.Variations().size();
    std::size_t rows = std::distance(pos_begin, pos_end);
    std::size_t cols = indexer.index_space_size;

    MatrixCSR<int> matrix{ elements_per_row, cols, rows };
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < static_cast<int64_t>(rows); i++)
        indexer.InsertIndices(*(pos_begin + i), matrix.Row(i));

    DiagonalPreconditioner P(matrix.JacobiPreconditionerSquare(100.0f));

    PCG solver(transposed(matrix) * matrix, P, model.Weights(), transposed(matrix) * score);
    solver.Iterate(iterations);

    model.SetWeights(solver.X());
}

void Fit(GLEM& model, const std::vector<Position>& pos, const std::vector<float>& score, int iterations)
{
    Fit(model, pos.begin(), pos.end(), score, iterations);
}

double Fit(AM& model, const std::vector<PositionMultiDepthScore>& data)
{
    struct XY {
        int D, d, e, score_diff;

        auto operator<=>(const XY&) const noexcept = default;

        std::vector<int> x() const { return { D, d, e }; }
        int y() const { return score_diff; }
    };

    std::vector<XY> xy;
    for (const PositionMultiDepthScore& datum : data)
        for (int big = 0; big < datum.score_of_depth.size(); big++)
            if (datum.score_of_depth[big] != undefined_score)
                for (int small = 0; small < big; small++)
                    if (datum.score_of_depth[small] != undefined_score)
                        xy.emplace_back(big, small, datum.pos.EmptyCount(), datum.score_of_depth[big] - datum.score_of_depth[small]);

    std::sort(std::execution::par, xy.begin(), xy.end());
    auto chunks = xy | ranges::views::chunk_by([](const XY& l, const XY& r) { return l.x() == r.x(); });
    auto x = chunks | ranges::views::transform([](auto&& rng) { return ranges::front(rng).x(); }) | ranges::to_vector;
    auto y = chunks | ranges::views::transform([](auto&& rng) { return StandardDeviation(rng | ranges::views::transform(&XY::y)); }) | ranges::to_vector;
    std::size_t x_size = std::ranges::size(x);
    std::size_t y_size = std::ranges::size(y);
    if (x_size != y_size)
        throw std::runtime_error("Size mismatch!");

    model.param_values = NonLinearLeastSquaresFit(model.Function(), model.Parameters(), model.Variables(), x, y, model.param_values, /*steps*/ 1'000, /*damping_factor*/ 0.1);

    std::vector<double> error(x_size);
    for (std::size_t i = 0; i < x_size; i++)
        error[i] = model.Eval(x[i]) - y[i];

    float R_sq = 1.0f - Variance(error) / Variance(y);
    return R_sq;
}

double Fit(AAGLEM& model,
    const std::vector<Game>& train_games, int exact_blocks, Intensity train_eval, 
    const std::vector<Game>& accuracy_fit_games, int accuracy_fit_eval_all_depth_till, int accuracy_fit_eval_max_depth)
{
    HT tt{ 10'000'000 };
    IDAB<PVS> alg{ tt, model };

    std::vector<Position> train = Positions(train_games);
    std::vector<Position> rest = Positions(accuracy_fit_games);
    std::vector<PositionMultiDepthScore> accuracy_fit(rest.begin(), rest.end());

    for (int block = 0; block < model.Blocks(); block++)
    {
        int empty_count_begin = block * model.BlockSize();
        int empty_count_end = (block + 1) * model.BlockSize();
        Intensity intensity = (block < exact_blocks) ? Intensity::Exact() : train_eval;

        std::vector<PosScore> train_block;
        for (int i = 0; i < train.size(); i++)
        {
            int empty_count = train[i].EmptyCount();
            if (empty_count_begin <= empty_count and empty_count < empty_count_end)
                train_block.emplace_back(train[i]);
        }

        alg.clear();
        #pragma omp parallel for
        for (int i = 0; i < static_cast<int>(train_block.size()); i++)
            train_block[i].score = alg.Eval(train_block[i].pos, intensity).score;

        Fit(model.Evaluator(block), train_block, /*iterations*/ 10);
    }

    if (accuracy_fit.empty())
        return 0;
    alg.clear();
    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(accuracy_fit.size()); i++)
    {
        int empty_count = accuracy_fit[i].pos.EmptyCount();
        for (int d = 0; d <= empty_count; d++)
            if (d <= accuracy_fit_eval_max_depth or empty_count <= accuracy_fit_eval_all_depth_till)
                accuracy_fit[i].score_of_depth[d] = alg.Eval(accuracy_fit[i].pos, d).score;
    }
    return Fit(model.AccuracyModel(), accuracy_fit);
}