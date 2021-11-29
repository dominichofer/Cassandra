#pragma once
#include "Core/HashTable.h"
#include "Core/Position.h"
#include <atomic>
#include <cassert>
#include <cstdint>

template <typename T, typename T::value_type locked_state = static_cast<T::value_type>(-1)>
class ScopedLock
{
	T& atomic;
public:
	typename T::value_type value;

	// locks
	ScopedLock(T& atomic) : atomic(atomic)
	{
		do { // atomic spinlock
			value = atomic.exchange(locked_state, std::memory_order_acquire);
		} while (value == locked_state);
	}

	// unlocks
	~ScopedLock()
	{
		assert(value != locked_state);
		atomic.store(value, std::memory_order_release);
	}
};

struct PositionDepthPair
{
	Position pos{};
	int depth = 0;

	auto operator<=>(const PositionDepthPair&) const noexcept = default;
};

class BigNode
{
public:
	using key_type = PositionDepthPair;
	using value_type = uint64_t;

	BigNode() noexcept = default;

	void Update(const key_type&, const value_type&);
	std::optional<value_type> LookUp(const key_type&) const;
	void Clear();
private:
	mutable std::atomic<value_type> m_value{0}; // used as value and lock.
	key_type m_key;

	struct KeyValuePair { key_type key; value_type value; };
	KeyValuePair Get() const; // thread-safe
};

static_assert(sizeof(BigNode) <= 64);


struct BigNodeHashTable : public HashTable<BigNode>
{
	BigNodeHashTable(uint64_t buckets)
		: HashTable(buckets,
			[](const HashTable::key_type& key) {
				uint64_t P = key.pos.Player();
				uint64_t O = key.pos.Opponent();
				P ^= P >> 36;
				O ^= O >> 21;
				return (P * O + key.depth);
			})
	{}
};
