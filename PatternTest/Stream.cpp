#include "pch.h"
#include "Pattern.h"

TEST(Stream, serialize_deserialize_ScoreEstimator)
{
	ScoreEstimator in(pattern_mix);
	std::vector<float> weights(in.WeightsSize());
	ranges::iota(weights, 1);
	in.Weights(weights);

	std::stringstream stream;
	Serialize(in, stream);
	auto out = Deserialize<decltype(in)>(stream);

	EXPECT_EQ(in.Pattern(), out.Pattern());
	EXPECT_EQ(in.Weights(), out.Weights());
}

TEST(Stream, serialize_deserialize_MSSE)
{
	MSSE in(10, pattern_mix);
	std::vector<float> weights(in.WeightsSize());
	ranges::iota(weights, 1);
	in.Weights(weights);

	std::stringstream stream;
	Serialize(in, stream);
	auto out = Deserialize<decltype(in)>(stream);

	EXPECT_EQ(in.StageSize(), out.StageSize());
	EXPECT_EQ(in.Pattern(), out.Pattern());
	EXPECT_EQ(in.Weights(), out.Weights());
}

TEST(Stream, serialize_deserialize_AM)
{
	AM in({ 1, 2, 3, 4, 5 });

	std::stringstream stream;
	Serialize(in, stream);
	auto out = Deserialize<decltype(in)>(stream);

	EXPECT_EQ(in.ParameterValues(), out.ParameterValues());
}

TEST(Stream, serialize_deserialize_AAMSSE)
{
	MSSE estimator(10, pattern_mix);
	std::vector<float> weights(estimator.WeightsSize());
	ranges::iota(weights, 1);
	estimator.Weights(weights);

	AM model({ 1, 2, 3, 4, 5 });

	AAMSSE in(estimator, model);

	std::stringstream stream;
	Serialize(in, stream);
	auto out = Deserialize<decltype(in)>(stream);

	EXPECT_EQ(in.score_estimator.StageSize(), out.score_estimator.StageSize());
	EXPECT_EQ(in.score_estimator.Pattern(), out.score_estimator.Pattern());
	EXPECT_EQ(in.score_estimator.Weights(), out.score_estimator.Weights());
	EXPECT_EQ(in.accuracy_estimator.ParameterValues(), out.accuracy_estimator.ParameterValues());
}