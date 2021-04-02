#include "pch.h"

using namespace Pattern;

constexpr BitBoard HorizontalSymmetric{ 0x00000000000000E7ULL };
constexpr BitBoard DiagonalSymmetric{ 0x8040201008040303ULL };
constexpr BitBoard Asymmetric{ 0x000000000000000FULL };
TEST(MetaTest, HorizontalSymmetric)
{
	ASSERT_EQ(HorizontalSymmetric, FlipHorizontal(HorizontalSymmetric));
}

TEST(MetaTest, DiagonalSymmetric)
{
	ASSERT_EQ(DiagonalSymmetric, FlipDiagonal(DiagonalSymmetric));
}

TEST(MetaTest, Asymmetric)
{
	ASSERT_NE(Asymmetric, FlipHorizontal(Asymmetric));
	ASSERT_NE(Asymmetric, FlipDiagonal(Asymmetric));
}

TEST(DenseIndexer, CreateDenseIndexer_HorizontalSymmetric)
{
	const auto indexer = CreateDenseIndexer(HorizontalSymmetric);
	ASSERT_EQ(indexer->variations, 4);

	std::vector<int> indices;
	for (const Position& pos : Configurations(HorizontalSymmetric))
		indices.push_back(indexer->DenseIndex(pos, 0));
	std::sort(indices.begin(), indices.end());

	for (int i = 0; i < indexer->reduced_size; i++)
		ASSERT_TRUE(std::find(indices.begin(), indices.end(), i) != indices.end());
}

TEST(DenseIndexer, CreateDenseIndexer_DiagonalSymmetric)
{
	const auto indexer = CreateDenseIndexer(DiagonalSymmetric);
	ASSERT_EQ(indexer->variations, 4);

	std::vector<int> indices;
	for (const Position& pos : Configurations(DiagonalSymmetric))
		indices.push_back(indexer->DenseIndex(pos, 0));
	std::sort(indices.begin(), indices.end());

	for (int i = 0; i < indexer->reduced_size; i++)
		ASSERT_TRUE(std::find(indices.begin(), indices.end(), i) != indices.end());
}

TEST(DenseIndexer, CreateDenseIndexer_Asymmetric)
{
	const auto indexer = CreateDenseIndexer(Asymmetric);
	ASSERT_EQ(indexer->variations, 8);

	std::vector<int> indices;
	for (const Position& pos : Configurations(Asymmetric))
		indices.push_back(indexer->DenseIndex(pos, 0));
	std::sort(indices.begin(), indices.end());

	for (int i = 0; i < indexer->reduced_size; i++)
		ASSERT_TRUE(std::find(indices.begin(), indices.end(), i) != indices.end());
}

TEST(DenseIndexer, CreateDenseIndexer_Group)
{
	const auto indexer = CreateDenseIndexer({HorizontalSymmetric, DiagonalSymmetric, Asymmetric});
	ASSERT_EQ(indexer->variations, 16);

	std::vector<int> indices;
	for (const Position& pos : Configurations(HorizontalSymmetric))
		indices.push_back(indexer->DenseIndex(pos, 0));
	for (const Position& pos : Configurations(DiagonalSymmetric))
		indices.push_back(indexer->DenseIndex(pos, 4));
	for (const Position& pos : Configurations(Asymmetric))
		indices.push_back(indexer->DenseIndex(pos, 8));

	for (int i = 0; i < indexer->reduced_size; i++)
		ASSERT_TRUE(std::find(indices.begin(), indices.end(), i) != indices.end());
}

void Test_SymmetryIndependance(const BitBoard pattern, const Weights& compressed)
{
	const auto eval = CreateEvaluator(pattern, compressed);

	// Assert score's independance of symmetry flips
	for (auto config : Configurations(pattern))
	{
		const auto score_0 = eval->Eval(config);
		config.FlipHorizontal();
		const auto score_1 = eval->Eval(config);
		config.FlipVertical();
		const auto score_2 = eval->Eval(config);
		config.FlipHorizontal();
		const auto score_3 = eval->Eval(config);
		config.FlipDiagonal();
		const auto score_4 = eval->Eval(config);
		config.FlipHorizontal();
		const auto score_5 = eval->Eval(config);
		config.FlipVertical();
		const auto score_6 = eval->Eval(config);
		config.FlipHorizontal();
		const auto score_7 = eval->Eval(config);

		ASSERT_EQ(score_0, score_1);
		ASSERT_EQ(score_0, score_2);
		ASSERT_EQ(score_0, score_3);
		ASSERT_EQ(score_0, score_4);
		ASSERT_EQ(score_0, score_5);
		ASSERT_EQ(score_0, score_6);
		ASSERT_EQ(score_0, score_7);
	}
}

TEST(Evaluator, HorizontalSymmetric_is_independant_of_symmetry)
{
	const auto weights = Weights(CreateDenseIndexer(HorizontalSymmetric)->reduced_size, 0);
	Test_SymmetryIndependance(HorizontalSymmetric, weights);
}

TEST(Evaluator, DiagonalSymmetric_is_independant_of_symmetry)
{
	const auto weights = Weights(CreateDenseIndexer(DiagonalSymmetric)->reduced_size, 0);
	Test_SymmetryIndependance(DiagonalSymmetric, weights);
}

TEST(Evaluator, Asymmetric_is_independant_of_symmetry)
{
	const auto weights = Weights(CreateDenseIndexer(Asymmetric)->reduced_size, 0);
	Test_SymmetryIndependance(Asymmetric, weights);
}

void Test_Expansion(BitBoard pattern)
{
	auto indexer = CreateDenseIndexer(pattern);

	Weights compressed(indexer->reduced_size);
	std::iota(compressed.begin(), compressed.end(), 1);
	auto evaluator = CreateEvaluator(pattern, compressed);
	PosGen::Random rnd(78);

	for (int i = 0; i < 100'000; i++)
	{
		Position pos = rnd();

		float sum = 0;
		for (int i = 0; i < indexer->variations; i++)
			sum += compressed[indexer->DenseIndex(pos, i)];

		ASSERT_EQ(sum, evaluator->Eval(pos));
	}
}

TEST(Evaluator, Weights_Expansion_HorizontalSymmetric)
{
	Test_Expansion(HorizontalSymmetric);
}

TEST(Evaluator, Weights_Expansion_DiagonalSymmetric)
{
	Test_Expansion(DiagonalSymmetric);
}

TEST(Evaluator, Weights_Expansion_Asymmetric)
{
	Test_Expansion(Asymmetric);
}
