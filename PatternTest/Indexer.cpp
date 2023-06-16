#include "pch.h"
#include "Pattern.h"
#include <set>

namespace
{
	template <typename T>
	void SymmetryIndependent(const T& indexer)
	{
		// Fuzzing
		RandomPositionGenerator gen(/*seed*/ 1337);
		for (int i = 0; i < 10'000; i++)
		{
			Position pos = gen();
			auto base_indices = indexer.Indices(pos);
			std::ranges::sort(base_indices);

			for (auto var : SymmetricVariants(pos))
			{
				auto var_indices = indexer.Indices(var);
				std::ranges::sort(var_indices);
				ASSERT_EQ(var_indices, base_indices);
			}
		}
	}

	void SymmetryIndependent(uint64_t pattern)
	{
		auto indexer = CreateIndexer(pattern);
		SymmetryIndependent(*indexer);
	}

	void SymmetryIndependent(std::vector<uint64_t> pattern)
	{
		auto indexer = GroupIndexer(pattern);
		SymmetryIndependent(indexer);
	}

	TEST(Indexer, A_symmetric_pattern_is_independent_of_symmetry_flips) { SymmetryIndependent(pattern_a); }
	TEST(Indexer, C_symmetric_pattern_is_independent_of_symmetry_flips) { SymmetryIndependent(pattern_c); }
	TEST(Indexer, D_symmetric_pattern_is_independent_of_symmetry_flips) { SymmetryIndependent(pattern_d); }
	TEST(Indexer, H_symmetric_pattern_is_independent_of_symmetry_flips) { SymmetryIndependent(pattern_h); }
	TEST(Indexer, V_symmetric_pattern_is_independent_of_symmetry_flips) { SymmetryIndependent(pattern_v); }
	TEST(Indexer, VH_symmetric_pattern_is_independent_of_symmetry_flips) { SymmetryIndependent(pattern_vh); }
	TEST(Indexer, DC_symmetric_pattern_is_independent_of_symmetry_flips) { SymmetryIndependent(pattern_dc); }
	TEST(Indexer, VHCD_symmetric_pattern_is_independent_of_symmetry_flips) { SymmetryIndependent(pattern_vhdc); }
	TEST(GroupIndexer, Mixed_patterns_are_independent_of_symmetry_flips) { SymmetryIndependent(pattern_mix); }


	// Tests that the indexer covers the index space when it is called with all configurations of the pattern.
	template <typename T>
	void ConfigurationsCoverIndexSpace(uint64_t pattern, const T& indexer)
	{
		// Collect index of all configurations.
		std::set<int> set;
		for (auto config : Configurations(pattern))
			for (int index : indexer.Indices(config))
				set.insert(index);

		// Assert that each index is in the set
		for (int i = 0; i < indexer.index_space_size; i++)
			ASSERT_TRUE(set.contains(i));
	}

	// Tests that the indexer covers the index space when it is called with all configurations of the pattern.
	void ConfigurationsCoverIndexSpace(uint64_t pattern)
	{
		auto indexer = CreateIndexer(pattern);
		ConfigurationsCoverIndexSpace(pattern, *indexer);
	}

	// Tests that the indexer covers the index space when it is called with all configurations of the pattern.
	void ConfigurationsCoverIndexSpace(std::vector<uint64_t> pattern)
	{
		uint64_t pattern_union = 0;
		for (uint64_t p : pattern)
			pattern_union |= p;

		auto indexer = GroupIndexer(pattern);

		ConfigurationsCoverIndexSpace(pattern_union, indexer);
	}

	TEST(Indexer, A_symmetric_pattern_covers_index_space) { ConfigurationsCoverIndexSpace(pattern_a); }
	TEST(Indexer, C_symmetric_pattern_covers_index_space) { ConfigurationsCoverIndexSpace(pattern_c); }
	TEST(Indexer, D_symmetric_pattern_covers_index_space) { ConfigurationsCoverIndexSpace(pattern_d); }
	TEST(Indexer, H_symmetric_pattern_covers_index_space) { ConfigurationsCoverIndexSpace(pattern_h); }
	TEST(Indexer, V_symmetric_pattern_covers_index_space) { ConfigurationsCoverIndexSpace(pattern_v); }
	TEST(Indexer, VH_symmetric_pattern_covers_index_space) { ConfigurationsCoverIndexSpace(pattern_vh); }
	TEST(Indexer, DC_symmetric_pattern_covers_index_space) { ConfigurationsCoverIndexSpace(pattern_dc); }
	TEST(Indexer, VHCD_symmetric_pattern_covers_index_space) { ConfigurationsCoverIndexSpace(pattern_vhdc); }
	TEST(GroupIndexer, Mixed_patterns_covers_index_space) { ConfigurationsCoverIndexSpace(pattern_mix); }
}