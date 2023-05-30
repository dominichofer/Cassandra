#include "pch.h"
#include "Pattern.h"

namespace
{
	void SymmetryIndependent(BitBoard pattern)
	{
		ScoreEstimator estimator(pattern);

		std::vector<float> weights(estimator.WeightsSize());
		ranges::iota(weights, 1);

		estimator.Weights(weights);
		ASSERT_EQ(weights, estimator.Weights());

		// Assert score's independance of symmetry flips
		for (auto config : Configurations(pattern))
		{
			auto base_score = estimator.Eval(config);
			for (auto var : SymmetricVariants(config))
				ASSERT_EQ(base_score, estimator.Eval(var));
		}
	}

	TEST(ScoreEstimator, A_symmetric_pattern_is_independent_of_symmetry_flips) { SymmetryIndependent(pattern_a); }
	TEST(ScoreEstimator, C_symmetric_pattern_is_independent_of_symmetry_flips) { SymmetryIndependent(pattern_c); }
	TEST(ScoreEstimator, D_symmetric_pattern_is_independent_of_symmetry_flips) { SymmetryIndependent(pattern_d); }
	TEST(ScoreEstimator, H_symmetric_pattern_is_independent_of_symmetry_flips) { SymmetryIndependent(pattern_h); }
	TEST(ScoreEstimator, V_symmetric_pattern_is_independent_of_symmetry_flips) { SymmetryIndependent(pattern_v); }
	TEST(ScoreEstimator, VH_symmetric_pattern_is_independent_of_symmetry_flips) { SymmetryIndependent(pattern_vh); }
	TEST(ScoreEstimator, DC_symmetric_pattern_is_independent_of_symmetry_flips) { SymmetryIndependent(pattern_dc); }
	TEST(ScoreEstimator, VHCD_symmetric_pattern_is_independent_of_symmetry_flips) { SymmetryIndependent(pattern_vhdc); }

	TEST(ScoreEstimator, DetailedEval_size_of_edax) { ASSERT_EQ(ScoreEstimator(pattern::edax).DetailedEval(Position{}).size(), 46); }
	TEST(ScoreEstimator, DetailedEval_size_of_logistello) { ASSERT_EQ(ScoreEstimator(pattern::logistello).DetailedEval(Position{}).size(), 46); }
}