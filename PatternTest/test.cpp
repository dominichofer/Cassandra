#include "pch.h"
#include "PatternIO/PatternIO.h"
#include <vector>

const BitBoard pattern_h =
	"- - - - - - - -"
	"- - - - - - - -"
	"- - - - - - - -"
	"- - - - - - - -"
	"- - - - - - - -"
	"- - - - - - - -"
	"- # - - - - # -"
	"# # # # # # # #"_BitBoard;

const BitBoard pattern_d =
	"- - - - - - - -"
	"- - - - - - - -"
	"- - - - - - - -"
	"- - - - - - - -"
	"- - - - # - - -"
	"- - - - - # - #"
	"- - - - - - # #"
	"- - - - - # # #"_BitBoard;

const BitBoard pattern_a =
	"- - - - - - - -"
	"- - - - - - - -"
	"- - - - - - - -"
	"- - - - - - - -"
	"- - - - - - - -"
	"- - - - - - - -"
	"- - - # # # # #"
	"- - - # # # # #"_BitBoard;

template <typename T>
std::set<T> SymmetryVariations(const T& t)
{
	return {
		t,
		FlipCodiagonal(t),
		FlipDiagonal(t),
		FlipHorizontal(t),
		FlipVertical(t),
		FlipCodiagonal(FlipHorizontal(t)),
		FlipDiagonal(FlipHorizontal(t)),
		FlipVertical(FlipHorizontal(t))
	};
}

TEST(Metatest, PatternH)
{
	ASSERT_EQ(pattern_h, FlipHorizontal(pattern_h));
}
TEST(Metatest, PatternD)
{
	ASSERT_EQ(pattern_d, FlipDiagonal(pattern_d));
}
TEST(Metatest, PatternA)
{
	ASSERT_NE(pattern_a, FlipHorizontal(pattern_a));
	ASSERT_NE(pattern_a, FlipVertical(pattern_a));
	ASSERT_NE(pattern_a, FlipDiagonal(pattern_a));
	ASSERT_NE(pattern_a, FlipCodiagonal(pattern_a));
}

TEST(Metatest, SymmetryVariations)
{
	ASSERT_EQ(SymmetryVariations(pattern_h).size(), std::size_t(4));
	ASSERT_EQ(SymmetryVariations(pattern_d).size(), std::size_t(4));
	ASSERT_EQ(SymmetryVariations(pattern_a).size(), std::size_t(8));
}

// Tests that the indexer covers the index space when it is called with all configurations of the pattern.
void ConfigurationsCoverIndexSpace(const Indexer& indexer, BitBoard pattern)
{
	std::vector<int> indices;
	for (auto config : Configurations(pattern))
		indices.push_back(indexer.DenseIndex(config, 0));

	for (int i = 0; i < indexer.index_space_size; i++)
	{
		auto is_i = [i](int value) { return value == i; };
		ASSERT_TRUE(ranges::any_of(indices, is_i));
	}
}

TEST(Indexer, HorizontalSymmetric)
{
	auto indexer = CreateIndexer(pattern_h);
	ASSERT_EQ(indexer->Variations().size(), 4);
	ConfigurationsCoverIndexSpace(*indexer, pattern_h);
}

TEST(Indexer, DiagonalSymmetric)
{
	auto indexer = CreateIndexer(pattern_d);
	ASSERT_EQ(indexer->Variations().size(), 4);
	ConfigurationsCoverIndexSpace(*indexer, pattern_d);
}

TEST(Indexer, Asymmetric)
{
	auto indexer = CreateIndexer(pattern_a);
	ASSERT_EQ(indexer->Variations().size(), 8);
	ConfigurationsCoverIndexSpace(*indexer, pattern_a);
}

TEST(GroupIndexer, Mix)
{
	GroupIndexer indexer({ pattern_h, pattern_d, pattern_a });

	auto size = indexer.Variations().size();
	auto size_h = CreateIndexer(pattern_h)->Variations().size();
	auto size_d = CreateIndexer(pattern_d)->Variations().size();
	auto size_a = CreateIndexer(pattern_a)->Variations().size();
	ASSERT_EQ(size, size_h + size_d + size_a);

	std::vector<int> tmp(size);
	std::vector<int> indices;
	for (auto config : Configurations(pattern_h))
	{
		indexer.InsertIndices(config, tmp);
		indices.push_back(tmp[0]);
	}
	for (auto config : Configurations(pattern_d))
	{
		indexer.InsertIndices(config, tmp);
		indices.push_back(tmp[size_h]);
	}
	for (auto config : Configurations(pattern_a))
	{
		indexer.InsertIndices(config, tmp);
		indices.push_back(tmp[size_h + size_d]);
	}

	// Verify that each index is present at least once
	ranges::sort(indices);
	EXPECT_EQ(indices.front(), 0);
	for (int i = 1; i < indexer.index_space_size; i++)
		EXPECT_GE(indices[i+1] - indices[i], 0);
	EXPECT_EQ(indices.back(), indexer.index_space_size - 1);
}

void SymmetryicIndependant(const BitBoard pattern, std::span<const float> weights)
{
	auto model = GLEM(pattern, weights);

	// Assert score's independance of symmetry flips
	for (auto config : Configurations(pattern))
	{
		auto original_score = model.Eval(config);
		for (auto var : SymmetryVariations(config))
			ASSERT_EQ(original_score, model.Eval(var));
	}
}

TEST(GLEM, HorizontalSymmetric_is_independent_of_symmetry)
{
	std::vector<float> weights(CreateIndexer(pattern_h)->index_space_size);
	ranges::iota(weights, 1);
	SymmetryicIndependant(pattern_h, weights);
}

TEST(GLEM, DiagonalSymmetric_is_independent_of_symmetry)
{
	std::vector<float> weights(CreateIndexer(pattern_d)->index_space_size);
	ranges::iota(weights, 1);
	SymmetryicIndependant(pattern_d, weights);
}

TEST(GLEM, Asymmetric_is_independent_of_symmetry)
{
	std::vector<float> weights(CreateIndexer(pattern_a)->index_space_size);
	ranges::iota(weights, 1);
	SymmetryicIndependant(pattern_a, weights);
}

void GLEM_Eval_is_equivalent_to_DenseIndex(BitBoard pattern)
{
	auto indexer = CreateIndexer(pattern);

	std::vector<float> weights(indexer->index_space_size);
	ranges::iota(weights, 1);
	auto model = GLEM(pattern, weights);
	PosGen::Random rnd(78);

	for (int i = 0; i < 100'000; i++)
	{
		Position pos = rnd();

		float sum = 0;
		for (int i = 0; i < indexer->Variations().size(); i++)
			sum += weights[indexer->DenseIndex(pos, i)];

		ASSERT_EQ(sum, model.Eval(pos));
	}
}

TEST(GLEM, Weights_Expansion_HorizontalSymmetric)
{
	GLEM_Eval_is_equivalent_to_DenseIndex(pattern_h);
}

TEST(GLEM, Weights_Expansion_DiagonalSymmetric)
{
	GLEM_Eval_is_equivalent_to_DenseIndex(pattern_d);
}

TEST(GLEM, Weights_Expansion_Asymmetric)
{
	GLEM_Eval_is_equivalent_to_DenseIndex(pattern_a);
}

TEST(Stream, serialize_deserialize_GLEM)
{
	std::vector pattern = {
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - # # # # #"
		"- - - # # # # #"_BitBoard,

		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - #"
		"- - - - - - # #"
		"- - - - - # # #"
		"- - - - # # # #"_BitBoard
	};
	GLEM in(pattern);

	std::stringstream stream;
	Serialize(in, stream);
	auto out = Deserialize<decltype(in)>(stream);

	EXPECT_TRUE(ranges::equal(in.Pattern(), out.Pattern()));
	EXPECT_TRUE(ranges::equal(in.Weights(), out.Weights()));
}

TEST(Stream, serialize_deserialize_AAGLEM)
{
	auto BB1 =
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - # # # # #"
		"- - - # # # # #"_BitBoard;

	auto BB2 =
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - -"
		"- - - - - - - #"
		"- - - - - - # #"
		"- - - - - # # #"
		"- - - - # # # #"_BitBoard;
	std::vector<int> block_boundaries = { 1,2,3 };
	std::valarray<double> acc = { 1,2,3,4,5 };
	AAGLEM in({ BB1, BB2 }, block_boundaries, acc);

	std::stringstream stream;
	Serialize(in, stream);
	auto out = Deserialize<decltype(in)>(stream);

	EXPECT_TRUE(ranges::equal(in.Pattern(), out.Pattern()));
	EXPECT_TRUE(ranges::equal(in.BlockBoundaries(), out.BlockBoundaries()));
	EXPECT_TRUE(ranges::equal(in.AccuracyModel().param_values, out.AccuracyModel().param_values));
	auto a = in.GetWeights();
	auto b = out.GetWeights();
	EXPECT_TRUE(ranges::equal(in.GetWeights(), out.GetWeights()));
}