#pragma once
#include <atomic>
#include <functional>
#include <vector>
#include <optional>
#include <new>
#include <type_traits>

// "Node" needs to provide:
// - typename key_type
// - typename value_type
// - void Update(const key_type&, const value_type&)
// - std::optional<value_type> LookUp(const key_type&)
// - void Clear()
template <typename Node>
class HashTable
{
	static_assert(std::is_default_constructible_v<Node>);
public:
	using node_type = Node;
	using key_type = Node::key_type;
	using value_type = Node::value_type;

	HashTable(std::size_t buckets, std::function<std::size_t(const key_type&)> hash_fkt)
		: hash_fkt(std::move(hash_fkt))
		, buckets(buckets)
	{}

	void Update(const key_type& key, const value_type& value)
	{
		updates++;
		auto index = hash_fkt(key) % buckets.size();
		buckets[index].Update(key, value);
	}

	std::optional<value_type> LookUp(const key_type& key) const
	{
		lookups++;
		auto index = hash_fkt(key) % buckets.size();
		const auto ret = buckets[index].LookUp(key);
		if (ret.has_value())
			++hits;
		return ret;
	}

	void Clear()
	{
		//updates = 0;
		//lookups = 0;
		//hits = 0;
		for (auto& bucket : buckets)
			bucket.Clear();
	}

	std::size_t Buckets() const { return buckets.size(); }
	std::size_t MemoryFootprint() const { return buckets.size() * sizeof(Node); }
	std::size_t UpdateCounter() const { return updates; }
	std::size_t LookUpCounter() const { return lookups; }
	std::size_t HitCounter() const { return hits; }
private:
	mutable std::atomic<uint64_t> updates{0}, lookups{0}, hits{0};
	std::function<std::size_t(const key_type&)> hash_fkt;
	/*alignas(std::hardware_destructive_interference_size)*/ std::vector<Node> buckets;
};