#include "ModelFit.h"
#include "PatternFit.h"
#include "Search/Puzzle.h"
#include "IO/IO.h"
#include "Math/Algorithm.h"
#include "Math/Statistics.h"
#include <map>
#include <vector>

void CrunchModelEvaluationPuzzles()
{
	PatternEval pattern_eval = DefaultPatternEval();
	HashTablePVS tt{ 100'000'000 };

	auto puzzles = Load<std::vector<Puzzle>>(R"(G:\Reversi\model_eval.puz)");
	std::reverse(puzzles.begin(), puzzles.end()); // TODO: Make this more elegant!
	Process(std::execution::par, puzzles,
		[&](Puzzle& p, std::size_t index) {
			if (p.pos.EmptyCount() > 30)
				return;
			for (int d = 0; d <= p.pos.EmptyCount() - 10; d++) {
				auto request = Request(d);
				p.insert(request);
				p.RemoveResult(request);
			}
			bool had_work = p.Solve(IDAB{ tt, pattern_eval });
			if (had_work) {
				#pragma omp critical
				std::cout << to_string(p) << '\n';
			}
		});
	std::reverse(puzzles.begin(), puzzles.end()); // TODO: Make this more elegant!
	Save(R"(G:\Reversi\model_eval.puz)", puzzles);
}

struct dDE { int d, D, E; auto operator<=>(const dDE&) const = default; };
struct VD { int depth, value; };

Vector to_Vector(const dDE& x)
{
	return { Vector::value_type(x.d), Vector::value_type(x.D), Vector::value_type(x.E) };
}

Vector Error(const SymExp& function, const Vars& vars,
	const std::vector<Vector>& x, const Vector& y)
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

void FitAccuracyModel()
{
	std::map<dDE, std::vector<int>> eval_diff;

	auto puzzles = Load<std::vector<Puzzle>>(R"(G:\Reversi\model_eval.puz)");
	for (const Puzzle& p : puzzles)
	{
		const int E = p.pos.EmptyCount();

		// Extract all data from puzzle
		std::vector<VD> vd;
		for (const Puzzle::Task& task : p.tasks)
			if (task.Intensity().IsCertain())
				vd.emplace_back(task.Intensity().depth, task.result.score);

		// Put pairwise score diffs into groups
		for (int i = 0; i < vd.size(); i++)
			for (int j = i + 1; j < vd.size(); j++)
			{
				int d = vd[i].depth;
				int D = vd[j].depth;
				int diff = vd[i].value - vd[j].value;
				if (d > D)
				{
					std::swap(d, D);
					diff = -diff;
				}

				eval_diff[{d, D, E}].push_back(diff);
			}
	}

	// Calculate standard deviation
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
	SymExp model = (exp(alpha * d) + beta) * pow(D - d, gamma) * (delta * E + epsilon);
	Vector initial_guess = { -0.2, 1, 0.25, 0, 1 };

	auto [R_sq, fitted_params] = FitModel(model, { alpha, beta, gamma, delta, epsilon }, { d, D, E }, x, y, initial_guess);

	std::cout << "Accuracy model params: ";
	for (auto p : fitted_params)
		std::cout << p << ", ";
	std::cout << "\n";
	std::cout << "R_sq: " << R_sq << std::endl;
}

void FitPattern()
{
	FitWeights();
	CrunchModelEvaluationPuzzles();
	FitAccuracyModel();
}
