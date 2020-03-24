#pragma once
#include "Core/Search.h"
#include "Core/HashTable.h"
#include <cstdint>
#include <atomic>
#include <optional>

using PVS_Info = Search::Result;

// TODO: Maybe this helps because it has a smaller memory footprint.
//struct PVS_Info
//{
//	// This class' size is critial
//
//	int8_t min = -Search::infinity;
//	int8_t max = +Search::infinity;
//	int8_t depth = -1;
//	uint8_t selectivity = 99;
//	uint8_t cost = 0;
//	CBestMoves best_moves{};
//
//	PvsInfo() = default;
//	PvsInfo(int8_t min, int8_t max, int8_t depth, uint8_t selectivity, CBestMoves, uint64_t node_count);
//};

class SpinlockMutex
{
	std::atomic_flag spinlock{};
public:
	void lock() { while (spinlock.test_and_set(std::memory_order_acquire)) continue; }
	void unlock() { spinlock.clear(std::memory_order_release); }
};

struct Node
{
	Position key =
		"X X X X X X X X"
		"X X X X X X X X"
		"X X X X X X X X"
		"X X X X X X X X"
		"X X X X X X X X"
		"X X X X X X X X"
		"X X X X X X X X"
		"X X X X X X X X"_pos;

	PVS_Info value{ ClosedInterval::Whole(), 0, Search::Selectivity::None, Field::invalid, 0 };
};

class OneNode
{
	mutable SpinlockMutex mutex{};
	Node node{};

public:
	OneNode() noexcept = default;
	OneNode(const OneNode&) noexcept;
	OneNode& operator=(const OneNode&) noexcept;

	void Update(Position, const PVS_Info&);
	std::optional<PVS_Info> LookUp(Position) const;
	void Clear();

	std::size_t NumberOfNonEmptyNodes() const;
};

// TODO: Implement TwoNode. It may help.

struct HashTablePVS : public HashTable<Position, PVS_Info, OneNode>
{
	HashTablePVS(uint64_t buckets) 
		: HashTable(buckets, 
			[](const Position& key)
			{ 
				uint64_t P = key.P;
				uint64_t O = key.O;
				P ^= P >> 36;
				O ^= O >> 21;
				return P * O; 
			})
	{}
};
