#pragma once
#include "Core/Core.h"
#include "Search/Search.h"
#include "Pattern/Pattern.h"
#include "Math/Math.h"
#include <valarray>

//namespace
//{
//    auto CreateMatrix(const GroupIndexer& indexer, const random_access_range<Position> auto& pos)
//    {
//        auto elements_per_row = indexer.Variations().size();
//        auto rows = ranges::size(pos);
//        auto cols = indexer.index_space_size;
//        MatrixCSR<int> mat(elements_per_row, cols, rows);
//
//        #pragma omp parallel for schedule(static)
//        for (int64_t i = 0; i < static_cast<int64_t>(rows); i++)
//            indexer.InsertIndices(pos[i], mat.Row(i));
//        return mat;
//    }
//
//    auto to_position()
//    {
//        return ranges::views::transform([](const NoMovePuzzle& p) { return p.pos; });
//    }
//
//    auto to_position(const ranges::range auto& puz)
//    {
//        return puz | to_position();
//    }
//
//    auto to_score()
//    {
//        return ranges::views::transform([](const NoMovePuzzle& p) { return static_cast<float>(p.ResultOf(p.MaxSolvedIntensity().value()).score); });
//    }
//
//    auto to_score(const ranges::range auto& puz)
//    {
//        return puz | to_score();
//    }
//
//    auto Error(const SymExp& function, const Vars& vars,
//        std::ranges::range auto&& x,
//        std::ranges::range auto&& y)
//    {
//        return ranges::views::transform(x, y,
//            [&function, &vars](const auto& x, const auto& y) {
//                return function.At(vars, x).value() - y;
//            });
//    }
//
//    auto FitModel(const SymExp& model, const Vars& params, const Vars& vars,
//        ranges::range auto&& x,
//        ranges::range auto&& y,
//        const std::valarray<double>& param_values)
//    {
//        static int counter = 0;
//        std::valarray<double> fitted_params = NonLinearLeastSquaresFit(model, params, vars, x, y, param_values);
//        auto fitted_model = model.At(params, fitted_params);
//        auto err = Error(fitted_model, vars, x, y);
//
//        std::fstream stream(fmt::format(R"(G:\Reversi\{}_err.log)", counter), std::ios::out);
//        stream << Variance(err) << "\n";
//        for (auto e : err)
//            stream << e << "\n";
//        stream.close();
//
//        stream = std::fstream(fmt::format(R"(G:\Reversi\{}_y.log)", counter++), std::ios::out);
//        stream << Variance(y) << "\n";
//        for (auto e : y)
//            stream << e << "\n";
//        stream.close();
//
//        float R_sq = 1.0 - Variance(err) / Variance(y);
//        return std::make_tuple(fitted_params, R_sq);
//    }
//}
//
//std::valarray<float> FitWeights(const std::ranges::random_access_range auto& data, const std::vector<BitBoard>& pattern, int iterations = 10)
//{
//    static int counter = 0;
//    auto indexer = GroupIndexer(pattern);
//    auto matrix = CreateMatrix(indexer, to_position(data));
//    std::valarray<float> score(ranges::size(data));
//    for (int i = 0; i < score.size(); i++)
//    {
//        auto p = static_cast<const NoMovePuzzle&>(data[i]);
//        score[i] = p.ResultOf(p.MaxSolvedIntensity().value()).score;
//    }
//    auto weights = std::valarray<float>(indexer.index_space_size);
//    DiagonalPreconditioner P(matrix.JacobiPreconditionerSquare(100.0f));
//    PCG solver(transposed(matrix) * matrix, P, weights, transposed(matrix) * score);
//    solver.Iterate(10);
//
//    std::fstream stream(fmt::format(R"(G:\Reversi\{}.log)", counter++), std::ios::out);
//    auto X = score;
//    auto Y = matrix * solver.X();
//    for (auto [x,y] : ranges::views::zip(X, Y))
//        stream << x << "," << y << "\n";
//    stream.close();
//
//    return solver.X();
//}
//template <ranges::range T>
//std::valarray<float> FitWeights_2(T&& data, const std::vector<BitBoard>& pattern, int iterations = 10)
//{
//    std::vector<ranges::range_value_t<T>> vec(data.begin(), data.end());
//    return FitWeights(vec, pattern, iterations);
//}
//template <ranges::range T>
//std::valarray<float> FitWeights_2(T&& data, auto&& add, const std::vector<BitBoard>& pattern, int iterations = 10)
//{
//    std::vector<ranges::range_value_t<T>> vec(data.begin(), data.end());
//    vec.insert(vec.end(), add.begin(), add.end());
//    return FitWeights(vec, pattern, iterations);
//}
//
//void EvalForAccuracyFit(range<NoMovePuzzle> auto& data, const Algorithm&, int max_depth);

// returns R^2
//double ImproveAccuracyModel(AAGLEM& model, ranges::range auto&& data)
//{
//    struct DepthScore { int depth, score; };
//    struct DDED {
//        int Depth, depth, empty_count, score_diff;
//        std::valarray<int> dde() const { return { Depth, depth, empty_count }; }
//        auto dde_tuple() const { return std::make_tuple(Depth, depth, empty_count); }
//    };
//
//    std::vector<DDED> tmp;
//    for (const NoMovePuzzle& p : data)
//    {
//        int empty_count = p.EmptyCount();
//        std::vector<DepthScore> relevant;
//        for (const NoMovePuzzle::Task& t : p.tasks)
//            if (t.IsDone() and t.IsCertain())
//                relevant.emplace_back(t.Depth(), t.Score());
//        for (const DepthScore& small : relevant)
//            for (const DepthScore& big : relevant)
//                if (small.depth < big.depth)
//                    tmp.emplace_back(big.depth, small.depth, empty_count, small.score - big.score);
//    }
//    std::sort(std::execution::par, tmp.begin(), tmp.end(), [](const DDED& l, const DDED& r) { return l.dde_tuple() < r.dde_tuple(); });
//    auto diffs = tmp | ranges::views::chunk_by([](const DDED& l, const DDED& r) { return l.dde_tuple() == r.dde_tuple(); });
//
//    auto x = diffs | ranges::views::transform([](auto&& rng) { return ranges::front(rng).dde(); });
//    auto y = diffs | ranges::views::transform([](auto&& rng) { return StandardDeviation(rng | ranges::views::transform(&DDED::score_diff)); });
//
//    Var D, d, E, alpha, beta, gamma, delta, epsilon;
//    SymExp accuracy_forumla = model.Accuracy(D, d, E, alpha, beta, gamma, delta, epsilon);
//
//    auto [fitted_params, R_sq] = FitModel(accuracy_forumla, { alpha, beta, gamma, delta, epsilon }, { D, d, E }, x, y, model.accuracy_parameters);
//    model.accuracy_parameters = fitted_params;
//    return R_sq;
//}
//
//void Improve(
//	AAGLEM&,
//	const range<NoMovePuzzle> auto& weight_fit_data,
//	range<NoMovePuzzle> auto& accuracy_model_data,
//	const Algorithm&,
//	int max_depth);


namespace
{
    auto CreateMatrix(const GroupIndexer& indexer, random_access_range<Position> auto&& pos)
    {
        auto elements_per_row = indexer.Variations().size();
        auto rows = ranges::size(pos);
        auto cols = indexer.index_space_size;
        MatrixCSR<int> mat(elements_per_row, cols, rows);

        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < static_cast<int64_t>(rows); i++)
            indexer.InsertIndices(pos[i], mat.Row(i));
        return mat;
    }

    template <std::ranges::random_access_range Rng>
    auto to_valarray(Rng&& rng) -> std::valarray<std::ranges::range_value_t<Rng>>
    {
        int64_t size = std::ranges::size(rng);
        std::valarray<std::ranges::range_value_t<decltype(rng)>> arr(size);
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < size; i++)
            arr[i] = rng[i];
        return arr;
    }
}

std::valarray<float> FittedWeights(const std::vector<BitBoard>& pattern, random_access_range<PosScore> auto&& data, int iterations = 10)
{
    auto pos = data | ranges::views::transform(&PosScore::pos);
    auto score = to_valarray(data | ranges::views::transform([](const PosScore& ps) { return static_cast<float>(ps.score); }));
    auto indexer = GroupIndexer(pattern);
    auto matrix = CreateMatrix(indexer, pos);
    auto weights = std::valarray<float>(indexer.index_space_size);
    DiagonalPreconditioner P(matrix.JacobiPreconditionerSquare(100.0f));
    PCG solver(transposed(matrix) * matrix, P, weights, transposed(matrix) * score);
    solver.Iterate(iterations);

    return solver.X();
}
std::valarray<float> FittedWeights(const std::vector<BitBoard>& pattern, range<PosScore> auto&& data, int iterations = 10)
{
    return FittedWeights(pattern, data | ranges::to_vector, iterations);
}

GLEM FittedGLEM(const std::vector<BitBoard>& pattern, random_access_range<PosScore> auto&& data, int iterations = 10)
{
    return { pattern, FittedWeights(pattern, data, iterations) };
}

std::valarray<double> FittedParameters(
    const SymExp& function,
    const Vars& params,
    const Vars& vars,
    ranges::random_access_range auto&& x,
    ranges::random_access_range auto&& y,
    const std::valarray<double>& param_values)
{
    return NonLinearLeastSquaresFit(function, params, vars, x, y, param_values, /*steps*/ 1'000, /*damping_factor*/ 0.1);
}

AM FittedAM(ranges::random_access_range auto&& x, ranges::random_access_range auto&& y)
{
    AM model;
    return { FittedParameters(model.Function(), model.Params(), model.Vars(), x, y, model.param_values) };
}

struct PosMultiDepthScore
{
    Position pos;
    std::array<int, 60> score_of_depth;

    PosMultiDepthScore(Position pos) : pos(pos) { score_of_depth.fill(undefined_score); }
};

AM FittedAM(range<PosMultiDepthScore> auto&& data)
{
    struct XY {
        int big_depth, small_depth, empty_count, score_diff;

        auto operator<=>(const XY&) const noexcept = default;

        std::valarray<int> x() const { return { big_depth, small_depth, empty_count }; }
        int y() const { return score_diff; }
    };

    std::vector<XY> xy;
    for (const PosMultiDepthScore& datum : data)
        for (int big = 0; big < datum.score_of_depth.size(); big++)
            if (datum.score_of_depth[big] != undefined_score)
                for (int small = 0; small < big; small++)
                    if (datum.score_of_depth[small] != undefined_score)
                        xy.emplace_back(big, small, datum.pos.EmptyCount(), datum.score_of_depth[big] - datum.score_of_depth[small]);

    std::sort(std::execution::par, xy.begin(), xy.end());
    auto chunks = xy | ranges::views::chunk_by([](const XY& l, const XY& r) { return ranges::all_of(l.x() == r.x(), std::identity()); });
    auto x = chunks | ranges::views::transform([](auto&& rng) { return ranges::front(rng).x(); }) | ranges::to_vector;
    auto y = chunks | ranges::views::transform([](auto&& rng) { return StandardDeviation(rng | ranges::views::transform(&XY::y)); }) | ranges::to_vector;

    return FittedAM(x, y);
}
