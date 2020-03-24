#include "pch.h"

using namespace Pattern;

constexpr BitBoard PatternH{ 0x00000000000000E7ULL }; // HorizontalSymmetric
constexpr BitBoard PatternD{ 0x8040201008040303ULL }; // DiagonalSymmetric
constexpr BitBoard PatternA{ 0x000000000000000FULL }; // Asymmetric
static const auto WeightsH = Weights(CreateIndexMapper(PatternH)->ReducedSize(), 0);
static const auto WeightsD = Weights(CreateIndexMapper(PatternD)->ReducedSize(), 0);
static const auto WeightsA = Weights(CreateIndexMapper(PatternA)->ReducedSize(), 0);

TEST(MetaTest, HorizontalSymmetric)
{
	ASSERT_EQ(PatternH, FlipHorizontal(PatternH));
}

TEST(MetaTest, DiagonalSymmetric)
{
	ASSERT_EQ(PatternD, FlipDiagonal(PatternD));
}

TEST(MetaTest, Asymmetric)
{
	ASSERT_NE(PatternA, FlipHorizontal(PatternA));
	ASSERT_NE(PatternA, FlipDiagonal(PatternA));
}

TEST(IndexMapper, CreateIndexMapper_HorizontalSymmetric)
{
	const auto im = CreateIndexMapper(PatternH);

	ASSERT_EQ(im->GroupOrder(), 4);
	ASSERT_EQ(im->Patterns()[0], PatternH);
	ASSERT_EQ(im->Indices(Position::Start()).size(), 4);
}

TEST(IndexMapper, CreateIndexMapper_DiagonalSymmetric)
{
	const auto im = CreateIndexMapper(PatternD);

	ASSERT_EQ(im->GroupOrder(), 4);
	ASSERT_EQ(im->Patterns()[0], PatternD);
	ASSERT_EQ(im->Indices(Position::Start()).size(), 4);
}

TEST(IndexMapper, CreateIndexMapper_Asymmetric)
{
	const auto im = CreateIndexMapper(PatternA);

	ASSERT_EQ(im->GroupOrder(), 8);
	ASSERT_EQ(im->Patterns()[0], PatternA);
	ASSERT_EQ(im->Indices(Position::Start()).size(), 8);
}

TEST(Evaluator, CreateEvaluator_works_for_horizontal_symmetric_pattern)
{
	const auto p = CreateEvaluator(PatternH, WeightsH);

	ASSERT_EQ(p->Pattern, PatternH);
}

TEST(Evaluator, CreateEvaluator_works_for_diagonal_symmetric_pattern)
{
	const auto p = CreateEvaluator(PatternD, WeightsD);

	ASSERT_EQ(p->Pattern, PatternD);
}

TEST(Evaluator, CreateEvaluator_works_for_asymmetric_pattern)
{
	const auto p = CreateEvaluator(PatternA, WeightsA);

	ASSERT_EQ(p->Pattern, PatternA);
}

void Test_SymmetryIndependance(const BitBoard pattern, const Weights& compressed)
{
	const auto eval = CreateEvaluator(pattern, compressed);

	// Assert score's independance of flips
	For_each_config(pattern,
		[&](Position pos) {
			const auto score = eval->Eval(pos);
			for (std::size_t i = 1; i < 8; i++)
			{
				switch (i)
				{
					case 1: pos.FlipHorizontal(); break;
					case 2: pos.FlipVertical(); break;
					case 3: pos.FlipHorizontal(); break;
					case 4: pos.FlipDiagonal(); break;
					case 5: pos.FlipHorizontal(); break;
					case 6: pos.FlipVertical(); break;
					case 7: pos.FlipHorizontal(); break;
				}
				auto other = eval->Eval(pos);
				ASSERT_EQ(score, other);
			}
		});
}

TEST(Evaluator, HorizontalSymmetric_is_independant_of_symmetry)
{
	Test_SymmetryIndependance(PatternH, WeightsH);
}

TEST(Evaluator, DiagonalSymmetric_is_independant_of_symmetry)
{
	Test_SymmetryIndependance(PatternD, WeightsD);
}

TEST(Evaluator, Asymmetric_is_independant_of_symmetry)
{
	Test_SymmetryIndependance(PatternA, WeightsA);
}

void Test_LegalWeights(BitBoard pattern)
{
	auto index_mapper = CreateIndexMapper(pattern);

	Weights compressed(index_mapper->ReducedSize());
	std::iota(compressed.begin(), compressed.end(), 1);
	auto evaluator = CreateEvaluator(pattern, compressed);
	PosGen::Random rnd(78);

	for (std::size_t i = 0; i < 100'000; i++)
	{
		const auto pos = rnd();

		auto configs = index_mapper->Indices(pos);
		float sum = 0;
		for (auto it : configs)
			sum += compressed[it];

		auto score = evaluator->Eval(pos);

		ASSERT_EQ(sum, score);
	}
}

TEST(Evaluator, LegalWeights_HorizontalSymmetric)
{
	Test_LegalWeights(PatternH);
}

TEST(Evaluator, LegalWeights_DiagonalSymmetric)
{
	Test_LegalWeights(PatternD);
}

TEST(Evaluator, LegalWeights_Asymmetric)
{
	Test_LegalWeights(PatternA);
}
