#pragma once
#include <atomic>
#include <functional>
#include <new>
#include <optional>
#include <type_traits>
#include <vector>

// "Node" needs to provide:
// - typename key_type
// - typename value_type
// - void Update(const key_type&, const value_type&)
// - std::optional<value_type> LookUp(const key_type&)
// - void clear()
template <typename Node>
class HashTable
{
#ifndef __NVCC__
	static_assert(std::is_default_constructible_v<Node>);
#endif
public:
	using node_type = Node;
	using key_type = Node::key_type;
	using value_type = Node::value_type;
private:
	mutable std::atomic<uint64_t> updates{0}, lookups{ 0 }, hits{ 0 };
	std::function<std::size_t(const key_type&)> hash_fkt;
	/*alignas(std::hardware_destructive_interference_size)*/ std::vector<Node> buckets;
public:
	HashTable(std::size_t buckets, std::function<std::size_t(const key_type&)> hash_fkt)
		: hash_fkt(std::move(hash_fkt))
		, buckets(buckets)
	{}
	HashTable(const HashTable<Node>& o)
		: updates(o.updates.load())
		, lookups(o.lookups.load())
		, hits(o.hits.load())
		, hash_fkt(o.hash_fkt)
		, buckets(o.buckets)
	{}
	HashTable(HashTable<Node>&& o)
		: updates(o.updates.load())
		, lookups(o.lookups.load())
		, hits(o.hits.load())
		, hash_fkt(o.hash_fkt)
		, buckets(std::move(o.buckets))
	{}
	HashTable<Node> operator=(const HashTable<Node>& o)
	{
		updates.store(o.updates.load());
		lookups.store(o.lookups.load());
		hits.store(o.hits.load());
		hash_fkt = o.hash_fkt;
		buckets = o.buckets;
		return *this;
	}
	HashTable<Node> operator=(HashTable<Node>&& o)
	{
		updates.store(o.updates.load());
		lookups.store(o.lookups.load());
		hits.store(o.hits.load());
		hash_fkt = o.hash_fkt;
		buckets = std::move(o.buckets);
		return *this;
	}

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
		auto ret = buckets[index].LookUp(key);
		if (ret.has_value())
			hits++;
		return ret;
	}

	void clear()
	{
		updates = 0;
		lookups = 0;
		hits = 0;
		for (auto& bucket : buckets)
			bucket.Clear();
	}

	std::size_t Buckets() const { return buckets.size(); }
	std::size_t UpdateCounter() const { return updates; }
	std::size_t LookUpCounter() const { return lookups; }
	std::size_t HitCounter() const { return hits; }
};