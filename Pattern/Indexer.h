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
	uint64_t pattern;
	int symmetry_group_order;
	int index_space_size = 0;

	Indexer(uint64_t pattern, int symmetry_group_order) noexcept;

	virtual int Index(Position) const = 0;
	std::vector<int> Indices(Position) const;
	virtual void InsertIndices(Position, std::span<int> location, int offset = 0) const = 0;
	virtual std::vector<uint64_t> Variations() const = 0;
};

std::unique_ptr<Indexer> CreateIndexer(uint64_t pattern);

std::size_t ConfigurationsOfPattern(uint64_t pattern);
std::size_t ConfigurationsOfPattern(std::vector<uint64_t> pattern);

class GroupIndexer
{
	std::vector<std::unique_ptr<Indexer>> indexers;
public:
	int index_space_size;

	GroupIndexer(std::vector<uint64_t> pattern);

	std::vector<int> Indices(Position) const;
	void InsertIndices(Position, std::span<int> location) const;
	std::vector<uint64_t> Variations() const;
};
