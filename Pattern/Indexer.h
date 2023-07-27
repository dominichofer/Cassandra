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

	virtual uint32_t Index(Position) const = 0;
	std::vector<uint32_t> Indices(Position) const;
	virtual void InsertIndices(Position, std::span<uint32_t> location, uint32_t offset = 0) const = 0;
	virtual std::vector<uint64_t> Variations() const = 0;
};

std::unique_ptr<Indexer> CreateIndexer(uint64_t pattern);

std::size_t ConfigurationCount(uint64_t pattern);
std::size_t ConfigurationCount(std::vector<uint64_t> pattern);

class GroupIndexer
{
	std::vector<std::unique_ptr<Indexer>> indexers;
public:
	int index_space_size;

	GroupIndexer(std::vector<uint64_t> pattern);

	std::vector<uint32_t> Indices(Position) const;
	void InsertIndices(Position, std::span<uint32_t> location) const;
	std::vector<uint64_t> Variations() const;
};
