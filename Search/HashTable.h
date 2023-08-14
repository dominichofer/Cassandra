#pragma once
#include "Game/Game.h"
#include "Result.h"
#include <atomic>
#include <optional>

class Spinlock
{
	std::atomic_flag locked{};
public:
	[[nodiscard]] Spinlock() noexcept = default;
	[[nodiscard]] bool try_lock() noexcept { return locked.test_and_set(std::memory_order_acquire); }
	void lock() noexcept { while (try_lock()) continue; }
	void unlock() noexcept { locked.clear(std::memory_order_release); }
};

class TranspositionValue
{
public:
	ClosedInterval<> window{ min_score, max_score };
	int8_t depth = -1;
	Field best_move = Field::PS;
	float confidence_level = 0.0f;

	TranspositionValue() noexcept = default;
	TranspositionValue(ClosedInterval<> window, int8_t depth, float confidence_level, Field best_move)
		: window(window), depth(depth), confidence_level(confidence_level), best_move(best_move)
	{}
	TranspositionValue(const Result& result)
		: TranspositionValue(result.Window(), result.depth, result.confidence_level, result.best_move)
	{}

	bool IsExact() const noexcept { return window.lower == window.upper; }
};

class OneNode
{
public:
	using key_type = Position;
	using value_type = TranspositionValue;
private:
	mutable Spinlock mutex{};
	key_type key{};
	value_type value{};
public:
	void Update(const key_type&, const value_type&);
	std::optional<value_type> LookUp(const key_type&) const;
	void Clear();
};

class TwoNodes
{
public:
	using key_type = Position;
	using value_type = TranspositionValue;
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
