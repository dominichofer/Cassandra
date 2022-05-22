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
	int variations, index_space_size;
	Indexer(BitBoard pattern, int variations, int index_space_size = 0) noexcept : pattern(pattern), variations(variations), index_space_size(index_space_size) {}

	virtual int DenseIndex(const Position&, int index) const = 0; // TODO: Rename to Index!
	virtual std::vector<BitBoard> Variations() const = 0;
	virtual void InsertIndices(const Position&, std::span<int> location, int offset) const = 0;
};

std::unique_ptr<Indexer> CreateIndexer(BitBoard pattern);

class GroupIndexer
{
public:
	std::vector<std::unique_ptr<Indexer>> indexers;
	int variations, index_space_size;

	GroupIndexer(const std::vector<BitBoard>& pattern);

	std::vector<BitBoard> Variations() const;
	void InsertIndices(const Position&, std::span<int> location) const;
};
