#pragma once
#include "Core/HashTable.h"
#include "Core/Position.h"
#include <atomic>
#include <cstdint>

struct PositionDepthPair
{
	Position position{ BitBoard{ 0 }, BitBoard{ 0 } }; // TODO: This is an illegal state of Position!
	int depth;
};

inline bool operator==(const PositionDepthPair& l, const PositionDepthPair& r) noexcept
{
	return (l.position == r.position) && (l.depth == r.depth);
}

// 
class BigNode
{
public:
	using key_type = PositionDepthPair;
	using value_type = uint64_t;

	BigNode() = default;
	BigNode(const BigNode&) = delete;
	BigNode(BigNode&&) = delete;
	BigNode& operator=(const BigNode&) = delete;
	BigNode& operator=(BigNode&&) = delete;
	~BigNode() = default;

	void Update(const key_type& new_key, const value_type new_value);

	std::optional<value_type> LookUp(const PositionDepthPair& key) const;

	void Clear();

private:
	class LockGuard
	{
		std::atomic<value_type>& lock;
	public:
		static inline value_type locked_marker = 0xFFFFFFFFFFFFFFFui64; // Reserved value to mark lock as locked.
		value_type value;

		LockGuard(std::atomic<value_type>& lock); // locks
		~LockGuard(); // unlocks
	};

	mutable std::atomic<value_type> m_value{ 0 }; // used as value and 
	key_type m_key;

	struct KeyValuePair { key_type key; value_type value; };
	KeyValuePair Get() const; // Thread safe access.
};

static_assert(sizeof(BigNode) <= std::hardware_constructive_interference_size);


struct BigNodeHashTable : public HashTable<BigNode::key_type, BigNode::value_type, BigNode>
{
	BigNodeHashTable(uint64_t buckets)
		: HashTable(buckets,
			[](const HashTable::key_type& key) {
				uint64_t P = key.position.GetP();
				uint64_t O = key.position.GetO();
				P ^= P >> 36;
				O ^= O >> 21;
				return (P * O + key.depth);
			})
	{}
};
