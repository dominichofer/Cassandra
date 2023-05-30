#pragma once
#include "Core/Core.h"
#include "Helpers.h"
#include <cstdint>
#include <memory>
#include <vector>
#include <span>

// Interface
struct Indexer
{
	BitBoard pattern;
	int symmetry_group_order; // mathematical order of the group
	int index_space_size = 0;

	Indexer(BitBoard pattern, int symmetry_group_order) noexcept : pattern(pattern), symmetry_group_order(symmetry_group_order) {}

	virtual int Index(Position) const = 0;
	std::vector<int> Indices(Position) const;
	virtual void InsertIndices(Position, std::span<int> location, int offset = 0) const = 0;
	virtual std::vector<BitBoard> Variations() const = 0;
};

std::unique_ptr<Indexer> CreateIndexer(BitBoard pattern);

class GroupIndexer
{
	std::vector<std::unique_ptr<Indexer>> indexers;
public:
	int index_space_size;

	GroupIndexer(std::vector<BitBoard> pattern);

	std::vector<int> Indices(Position) const;
	void InsertIndices(Position, std::span<int> location) const;
	std::vector<BitBoard> Variations() const;
};
