#include "PatternFit.h"
#include "Search/Puzzle.h"
#include "IO/IO.h"
#include <ranges>
#include <iostream>
#include <vector>
#include <map>

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

void WeightFitter::Fit()
{
    auto indexer = CreateDenseIndexer(pattern);
    { // train
        auto matrix = CreateMatrix(*indexer, data.train.Pos());
        weights = std::vector<float>(indexer->reduced_size, 0);
        DiagonalPreconditioner P(matrix.JacobiPreconditionerSquare(1000));
        PCG solver(transposed(matrix) * matrix, P, weights, transposed(matrix) * data.train.Score());
        solver.Iterate(10);
        weights = solver.X();
        train_error = data.train.Score() - matrix * weights;
    }
    { // test
        auto matrix = CreateMatrix(*indexer, data.test.Pos());
        test_error = data.test.Score() - matrix * weights;
    }
}

//std::tuple<double, double> FitWeights(int block)
//{
//    DataBase<Puzzle> db;
//    db.Add(R"(G:\Reversi\rnd\eval_fit.puz)");
//    db.Add(R"(G:\Reversi\play33\eval_fit.puz)");
//    const auto bs = AAGLEM::block_size;
//
//    auto filtered_db = db | std::views::filter([block, bs](const Puzzle& p) { return 1 + bs * block <= p.pos.EmptyCount() and p.pos.EmptyCount() <= bs * (block + 1);  });
//    WeightFitter fitter(LoadPattern(), filtered_db, 0.01 /*test_fraction*/);
//    fitter.Fit();
//    SaveWeights(fitter.Weights(), block);
//    return std::make_tuple(StandardDeviation(fitter.TrainError()), StandardDeviation(fitter.TestError()));
//}


void FitWeights(const DataBase<Puzzle>& data, std::vector<BitBoard> pattern, int block_size, int block, bool print_results)
{
    auto IsInBlock = [block, block_size](const Puzzle& p) {
        int E = p.pos.EmptyCount();
        int lowest_E = 1 + block_size * block;
        return (lowest_E <= E) and (E < lowest_E + block_size);
    };

    WeightFitter fitter(pattern, data | std::views::filter(IsInBlock), 0.01 /*test_fraction*/);
    fitter.Fit();
    SaveWeights(fitter.Weights(), block);
    if (print_results)
        std::cout << std::format("Block {}: train error {:.2f}, test error {:.2f}\n", block, StandardDeviation(fitter.TrainError()), StandardDeviation(fitter.TestError()));
}

void FitWeights(const DataBase<Puzzle>& data, std::vector<BitBoard> pattern, int block_size, bool print_results)
{
    for (int block = 0; block < 50 / block_size; block++)
        FitWeights(data, pattern, block_size, block, print_results);
}

void FitWeights(const DataBase<Puzzle>& data, int block, bool print_results)
{
    auto eval = DefaultPatternEval();
    FitWeights(data, eval.Pattern(), eval.block_size, block, print_results);
}

void FitWeights(const DataBase<Puzzle>& data, bool print_results)
{
    auto eval = DefaultPatternEval();
    FitWeights(data, eval.Pattern(), eval.block_size, print_results);
}

void EvaluateAccuracyFit(DataBase<Puzzle>& data, HashTablePVS& tt, AAGLEM& evaluator)
{
    Process(std::execution::par,
        data | std::views::reverse | std::views::filter([](const Puzzle& p) { return p.pos.EmptyCount() <= 25 and p.pos.EmptyCount() > 10; }),
        [&](Puzzle& p, std::size_t index)
        {
            p.erase_if([&p](const Puzzle::Task& t) { return t.Request() != Request::ExactScore(p.pos); });
            for (int d = 0; d <= std::min(10, p.pos.EmptyCount() - 10); d++)
                p.insert(Request(d));
            p.insert(Request::ExactScore(p.pos));

            p.Solve(IDAB{ tt, evaluator });
        });
}

struct dDE { int d, D, E; auto operator<=>(const dDE&) const = default; };
struct DV { int depth, value; };

Vector to_Vector(const dDE& x)
{
    return { Vector::value_type(x.d), Vector::value_type(x.D), Vector::value_type(x.E) };
}

Vector Error(const SymExp& function, const Vars& vars, const std::vector<Vector>& x, const Vector& y)
{
    Vector err(x.size());
    for (std::size_t i = 0; i < x.size(); i++)
        err[i] = function.At(vars, x[i]).value() - y[i];
    return err;
}

auto FitModel(const SymExp& model, const Vars& params, const Vars& vars,
    const std::vector<Vector>& x, const Vector& y,
    Vector params_values)
{
    auto fitted_params = NonLinearLeastSquaresFit(model, params, vars, x, y, params_values);
    auto fitted_model = model.At(params, fitted_params);
    auto err = Error(fitted_model, vars, x, y);

    float R_sq = 1.0f - Variance(err) / Variance(y);
    return std::make_tuple(R_sq, fitted_params);
}

void FitAccuracyModel(const DataBase<Puzzle>& data, const AAGLEM& evaluator)
{
    std::map<dDE, std::vector<int>> eval_diff;
    for (const Puzzle& p : data)
    {
        const int E = p.pos.EmptyCount();

        // Extract all data from puzzle
        std::vector<DV> dv;
        for (const Puzzle::Task& task : p.tasks)
            if (task.IsDone() and task.GetIntensity().IsCertain())
                dv.emplace_back(task.GetIntensity().depth, task.Score());

        // Put pairwise score diffs into groups
        for (int i = 0; i < dv.size(); i++)
            for (int j = i + 1; j < dv.size(); j++)
            {
                int d = dv[i].depth;
                int D = dv[j].depth;
                int diff = dv[i].value - dv[j].value;
                if (d > D)
                {
                    std::swap(d, D);
                    diff = -diff;
                }

                eval_diff[{d, D, E}].push_back(diff);
            }
    }

    // Calculate standard deviations
    std::map<Vector, float> eval_SD;
    for (const auto& [key, diffs] : eval_diff)
        eval_SD[to_Vector(key)] = StandardDeviation(diffs);

    // Transform to Vectors
    std::vector<Vector> x;
    Vector y;
    x.reserve(eval_SD.size());
    y.reserve(eval_SD.size());
    for (const auto& sd : eval_SD) {
        x.push_back(sd.first);
        y.push_back(sd.second);
    }

    Var d{ "d" }, D{ "D" }, E{ "E" };
    Var alpha, beta, gamma, delta, epsilon;
    SymExp model = evaluator.Accuracy(D, d, E, alpha, beta, gamma, delta, epsilon);
    Vector initial_guess = { -0.2, 1, 0.25, 0, 1 };

    auto [R_sq, fitted_params] = FitModel(model, { alpha, beta, gamma, delta, epsilon }, { d, D, E }, x, y, initial_guess);
    SaveModelParameters(fitted_params);

    std::cout << "Accuracy model params: ";
    for (auto p : fitted_params)
        std::cout << p << ", ";
    std::cout << "R_sq: " << R_sq << std::endl;
}

void FitPattern(const DataBase<Puzzle>& eval_fit, DataBase<Puzzle>& accuracy_fit, HashTablePVS& tt, AAGLEM& evaluator, std::vector<BitBoard> pattern, int block_size)
{
    FitWeights(eval_fit, pattern, block_size, true);
    EvaluateAccuracyFit(accuracy_fit, tt, evaluator);
    FitAccuracyModel(accuracy_fit, evaluator);
}

void FitPattern(const DataBase<Puzzle>& eval_fit, DataBase<Puzzle>& accuracy_fit, HashTablePVS& tt, AAGLEM& evaluator)
{
    FitPattern(eval_fit, accuracy_fit, tt, evaluator, evaluator.Pattern(), evaluator.block_size);
}
