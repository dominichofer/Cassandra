#include "Hashtable.h"
#include <cassert>

BigNode::LockGuard::LockGuard(std::atomic<value_type>& lock) : lock(lock)
{
	// Spinlock
	do
	{
		value = lock.exchange(locked_marker, std::memory_order_acquire);

	} while (value == locked_marker);

}

BigNode::LockGuard::~LockGuard()
{
	lock.store(value, std::memory_order_release);
}

void BigNode::Update(const key_type& new_key, const value_type new_value)
{
	assert(new_value != LockGuard::locked_marker);

	LockGuard lock_guard(m_value);
	if (new_value > lock_guard.value)
	{
		m_key = new_key;
		lock_guard.value = new_value;
	}
}

std::optional<BigNode::value_type> BigNode::LookUp(const PositionDepthPair& key) const
{
	const auto pair = Get();
	if (pair.key == key)
		return pair.value;
	return {};
}

void BigNode::Clear()
{
	LockGuard lock_guard(m_value);
	m_key = { { BitBoard{ 0 }, BitBoard{ 0 } }, 0 }; // TODO: This is an illegal state of Position!
	lock_guard.value = 0;
}

BigNode::KeyValuePair BigNode::Get() const
{
	LockGuard lock_guard(m_value);
	return { m_key, lock_guard.value };
}
