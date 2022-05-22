#pragma once
#include "Core/Core.h"
#include "Algorithm.h"
#include <atomic>
#include <optional>

class Spinlock
{
	std::atomic_flag locked{};
public:
	[[nodiscard]] Spinlock() = default;
	[[nodiscard]] bool try_lock() noexcept { return locked.test_and_set(std::memory_order_acquire); }
	void lock() noexcept { while (try_lock()) continue; }
	void unlock() noexcept { locked.clear(std::memory_order_release); }
};


class OneNode
{
public:
	using key_type = Position;
	using value_type = TT_Info;
private:
	mutable Spinlock mutex{};
	key_type key{};
	value_type value{};
public:
	OneNode() noexcept = default;

	void Update(const key_type&, const value_type&);
	std::optional<value_type> LookUp(const key_type&) const;
	void Clear();
};

static_assert(sizeof(OneNode) <= std::hardware_constructive_interference_size);

class TwoNodes
{
public:
	using key_type = Position;
	using value_type = TT_Info;
private:
	mutable Spinlock mutex{};
	key_type key1{}, key2{};
	value_type value1{}, value2{};
public:
	TwoNodes() noexcept = default;

	void Update(const key_type&, const value_type&);
	std::optional<value_type> LookUp(const key_type&) const;
	void Clear();
};

//static_assert(sizeof(TwoNodes) <= std::hardware_constructive_interference_size);


class HT : public HashTable<OneNode>
{
public:
	HT(std::size_t buckets)
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
