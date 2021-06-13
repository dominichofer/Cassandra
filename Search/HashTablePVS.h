#pragma once
#include "Core/Core.h"
#include "Objects.h"
#include <cstdint>
#include <atomic>
#include <optional>

class Spinlock
{
	std::atomic_flag spinlock{};
public:
	[[nodiscard]] bool try_lock() noexcept { return spinlock.test_and_set(std::memory_order_acquire); }
	void lock() noexcept { while (try_lock()) continue; }
	void unlock() noexcept { spinlock.clear(std::memory_order_release); }
};

struct TT_Info
{
	Search::Result result{ {0, Confidence{0}}, ClosedInterval::Whole() };
	Field best_move = Field::invalid;
	Field best_move_2 = Field::invalid;

	TT_Info() noexcept = default;
	TT_Info(const Search::Result& result, Field best_move) noexcept
		: result(result), best_move(best_move) {}

	[[nodiscard]] operator Search::Result() const noexcept { return result; }
};

class OneNode
{
public:
	using key_type = Position;
	using value_type = TT_Info;

	OneNode() noexcept = default;

	void Update(const key_type&, const value_type&);
	[[nodiscard]] std::optional<value_type> LookUp(const key_type&) const;
	void Clear();

private:
	mutable Spinlock mutex{};
	key_type key{};
	value_type value{};
};

static_assert(sizeof(OneNode) <= std::hardware_constructive_interference_size);

//class TwoNode
//{
//public:
//	using key_type = Position;
//	using value_type = TT_Info;
//
//	TwoNode() noexcept = default;
//
//	void Update(const key_type&, const value_type&);
//	std::optional<value_type> LookUp(const key_type&) const;
//	void Clear();
//
//private:
//	mutable Spinlock mutex{};
//	key_type key{};
//	value_type value{};
//};
//
//static_assert(sizeof(TwoNode) <= std::hardware_constructive_interference_size);

// TODO: Implement TwoNode. It might help.

class HashTablePVS : public HashTable<OneNode>
{
public:
	HashTablePVS(std::size_t buckets)
		: HashTable(buckets, 
			[](const Position& key) noexcept
			{ 
				uint64_t P = key.Player();
				uint64_t O = key.Opponent();
				P ^= P >> 36;
				O ^= O >> 21;
				return P * O; 
			})
	{}
};
