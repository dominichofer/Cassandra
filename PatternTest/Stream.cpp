#include "pch.h"
//#include "Pattern.h"
//
//TEST(Stream, serialize_deserialize_ScoreEstimator)
//{
//	ScoreEstimator in(pattern_mix);
//	std::vector<float> weights(in.WeightsSize());
//	ranges::iota(weights, 1);
//	in.Weights(weights);
//
//	std::stringstream stream;
//	Serialize(in, stream);
//	auto out = Deserialize<decltype(in)>(stream);
//
//	EXPECT_EQ(in.Pattern(), out.Pattern());
//	EXPECT_EQ(in.Weights(), out.Weights());
//}
//
//TEST(Stream, serialize_deserialize_MSSE)
//{
//	MSSE in(10, pattern_mix);
//	std::vector<float> weights(in.WeightsSize());
//	ranges::iota(weights, 1);
//	in.Weights(weights);
//
//	std::stringstream stream;
//	Serialize(in, stream);
//	auto out = Deserialize<decltype(in)>(stream);
//
//	EXPECT_EQ(in.StageSize(), out.StageSize());
//	EXPECT_EQ(in.Pattern(), out.Pattern());
//	EXPECT_EQ(in.Weights(), out.Weights());
//}
//
//TEST(Stream, serialize_deserialize_AM)
//{
//	AM in({ 1, 2, 3, 4, 5 });
//
//	std::stringstream stream;
//	Serialize(in, stream);
//	auto out = Deserialize<decltype(in)>(stream);
//
//	EXPECT_EQ(in.ParameterValues(), out.ParameterValues());
//}
//
//TEST(Stream, serialize_deserialize_PatternBasedEstimator)
//{
//	MSSE estimator(10, pattern_mix);
//	std::vector<float> weights(estimator.WeightsSize());
//	ranges::iota(weights, 1);
//	estimator.Weights(weights);
//
//	AM model({ 1, 2, 3, 4, 5 });
//
//	PatternBasedEstimator in(estimator, model);
//
//	std::stringstream stream;
//	Serialize(in, stream);
//	auto out = Deserialize<decltype(in)>(stream);
//
//	EXPECT_EQ(in.score.StageSize(), out.score.StageSize());
//	EXPECT_EQ(in.score.Pattern(), out.score.Pattern());
//	EXPECT_EQ(in.score.Weights(), out.score.Weights());
//	EXPECT_EQ(in.accuracy.ParameterValues(), out.accuracy.ParameterValues());
//}