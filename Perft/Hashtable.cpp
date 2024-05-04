#include "Hashtable.h"

uint64_t Hash(const Position& pos, int depth)
{
	const uint64_t kMul = 0x9ddfea08eb382d69ULL;
	uint64_t a = pos.Player() * kMul + depth;
	a ^= (a >> 47);
	uint64_t b = (pos.Opponent() ^ a) * kMul;
	b ^= (b >> 47);
	return b;
}

void HashTable::Insert(const Position& pos, int depth, uint64_t value)
{
	auto index = Hash(pos, depth);
	std::unique_lock lock(mutexes[index & 0xFFULL]);
	Bucket& bucket = buckets[index % buckets.size()];
	if (value > bucket.value)
		bucket = { pos, depth, value };
}

uint64_t HashTable::LookUp(const Position& pos, int depth) const
{
	auto index = Hash(pos, depth);
	std::shared_lock lock(mutexes[index & 0xFFULL]);
	const Bucket& bucket = buckets[index % buckets.size()];
	if (bucket.pos == pos and bucket.depth == depth)
		return bucket.value;
	return 0;
}

void HashTable::Clear()
{
	#pragma omp parallel for
	for (int64_t i = 0; i < static_cast<int64_t>(buckets.size()); i++)
		buckets[i] = {};
}
