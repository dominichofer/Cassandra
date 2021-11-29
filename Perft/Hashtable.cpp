#include "Hashtable.h"

void BigNode::Update(const key_type& new_key, const value_type& new_value)
{
	ScopedLock lock{m_value};
	if (new_value > lock.value)
	{
		m_key = new_key;
		lock.value = new_value;
	}
}

std::optional<BigNode::value_type> BigNode::LookUp(const key_type& key) const
{
	const auto pair = Get();
	if (pair.key == key)
		return pair.value;
	return {};
}

void BigNode::Clear()
{
	ScopedLock lock{m_value};
	m_key = key_type{};
	lock.value = 0;
}

BigNode::KeyValuePair BigNode::Get() const
{
	ScopedLock lock{m_value};
	return { m_key, lock.value };
}
