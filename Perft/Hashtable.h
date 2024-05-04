#pragma once
#include "Board/Board.h"
#include <cstdint>
#include <shared_mutex>

uint64_t Hash(const Position&, int depth);

struct Bucket
{
	Position pos{ 0, 0 };
	int depth = 0;
	uint64_t value = 0;
};

class HashTable
{
	mutable std::array<std::shared_mutex, 256> mutexes;
	std::vector<Bucket> buckets;
public:
	HashTable(std::size_t size) : buckets(size) {}

	void Insert(const Position& pos, int depth, uint64_t value);
	uint64_t LookUp(const Position& pos, int depth) const;
	void Clear();
};
