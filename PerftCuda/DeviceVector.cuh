#pragma once
#include "Chronosity.h"
#include <cstdint>
#include <utility>
#include <stdexcept>
#include <vector>
#include <algorithm>
#include <cassert>

// page-locked host memory
template <typename>
class PinnedVector;

// device memory
template <typename T>
class DeviceVector
{
	T* m_vec = nullptr;
	std::size_t m_size = 0;
	std::size_t m_capacity = 0;
public:
	DeviceVector() = default;
	DeviceVector(DeviceVector<T>&& o) noexcept { swap(o); }
	DeviceVector(const DeviceVector<T>& o, chronosity c = syn) : DeviceVector(o.size()) { assign(o, c); }
	DeviceVector(const PinnedVector<T>& o, chronosity c = syn) : DeviceVector(o.size()) { assign(o, c); }
	explicit DeviceVector(const std::vector<T>& o) : DeviceVector(o.size()) { assign(o); }
	explicit DeviceVector(std::size_t count) : m_size(count), m_capacity(count) { ErrorCheck([&](){ return cudaMalloc(&m_vec, count * sizeof(T)); }); }
	~DeviceVector() { ErrorCheck([&](){ return cudaFree(m_vec); }); }

	DeviceVector<T>& operator=(DeviceVector<T>&& o) noexcept { swap(o); }
	DeviceVector<T>& operator=(const DeviceVector<T>& o) { store(o, syn); return *this; }
	DeviceVector<T>& operator=(const PinnedVector<T>& o) { store(o, syn); return *this; }
	DeviceVector<T>& operator=(const std::vector<T>& o) { store(o); return *this; }

	bool operator==(const DeviceVector<T>&) const = delete; // Too expensive!
	bool operator!=(const DeviceVector<T>&) const = delete; // Too expensive!

	// Assigns data. Requires sufficient capacity.
	void assign(const PinnedVector<T>&, chronosity, cudaStream_t = 0);
	void assign(const DeviceVector<T>&, chronosity);
	void assign(const std::vector<T>&);

	// Stores data. Allocates memory if needed.
	void store(const PinnedVector<T>&, chronosity, cudaStream_t = 0);
	void store(const DeviceVector<T>&, chronosity);
	void store(const std::vector<T>&);

	std::vector<T> load() const;

	      T* data()       { return m_vec; }
	const T* data() const { return m_vec; }
	      T* begin()       { return m_vec; }
	const T* begin() const { return m_vec; }
	const T* cbegin() const { return m_vec; }
	      T* end()       { return m_vec + m_size; }
	const T* end() const { return m_vec + m_size; }
	const T* cend() const { return m_vec + m_size; }

	bool empty() const { return m_size == 0; }
	std::size_t size() const { return m_size; }
	std::size_t capacity() const { return m_capacity; }
	void reserve(std::size_t new_capacity, chronosity, cudaStream_t = 0);

	void Clear() { m_size = 0; }
	void resize(std::size_t count, chronosity c, cudaStream_t stream = 0) { reserve(count, c, stream); m_size = count; }
	void swap(DeviceVector<T>& o);
	void swap(DeviceVector<T>&& o);
};

template <typename T>
void swap(DeviceVector<T>& l, DeviceVector<T>& r) { l.swap(r); }


// View on a vector in cuda memory
template <typename T>
class DeviceVectorView
{
	T* m_vec = nullptr;
	std::size_t m_size = 0;
	std::size_t m_capacity = 0;
public:
	DeviceVectorView(DeviceVector<T>& o) noexcept : m_vec(o.data()), m_size(o.size()), m_capacity(o.capacity()) {}
	DeviceVectorView(const DeviceVectorView<T>&) = default;
	DeviceVectorView(DeviceVectorView<T>&&) noexcept = default;
	DeviceVectorView<T>& operator=(const DeviceVectorView<T>&) = default;
	DeviceVectorView<T>& operator=(DeviceVectorView<T>&&) noexcept = default;
	~DeviceVectorView() = default;

	bool operator==(const DeviceVectorView<T>&) const noexcept = delete; // Too expensive!
	bool operator!=(const DeviceVectorView<T>&) const noexcept = delete; // Too expensive!

	__device__       T& operator[](std::size_t index)       { return m_vec[index]; }
	__device__ const T& operator[](std::size_t index) const { return m_vec[index]; }
	__device__       T& at(std::size_t index);
	__device__ const T& at(std::size_t index) const;
	__device__       T& front()       { return m_vec[0]; }
	__device__ const T& front() const { return m_vec[0]; }
	__device__       T& back()       { return m_vec[m_size - 1]; }
	__device__ const T& back() const { return m_vec[m_size - 1]; }
	__device__       T* data()       { return m_vec; }
	__device__ const T* data() const { return m_vec; }
	__device__       T* begin()       { return m_vec; }
	__device__ const T* begin() const { return m_vec; }
	__device__ const T* cbegin() const { return m_vec; }
	__device__       T* end()       { return m_vec + m_size; }
	__device__ const T* end() const { return m_vec + m_size; }
	__device__ const T* cend() const { return m_vec + m_size; }

	__device__ bool empty() const { return m_size == 0; }
	__device__ std::size_t size() const { return m_size; }
	__device__ std::size_t capacity() const { return m_capacity; }

	__device__ void Clear() { m_size = 0; }
	__device__ void push_back(const T&);
	__device__ void push_back(T&&);
	__device__ void pop_back() { m_size--; }
	__device__ void swap(DeviceVectorView<T>&);
	__device__ void swap(DeviceVectorView<T>&&);
};

template <typename T>
__device__ void swap(DeviceVectorView<T>& l, DeviceVectorView<T>& r) { l.swap(r); }


///////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Implementation
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void DeviceVector<T>::assign(const PinnedVector<T>& src, chronosity c, cudaStream_t stream)
{
	assert(m_capacity >= src.size());

	if /*constexpr*/ (c == chronosity::syn)
		ErrorCheck([&]() { return cudaMemcpy(m_vec, src.data(), src.size() * sizeof(T), cudaMemcpyHostToDevice); });
	else
		ErrorCheck([&]() { return cudaMemcpyAsync(m_vec, src.data(), src.size() * sizeof(T), cudaMemcpyHostToDevice, stream); });

	m_size = src.size();
}

template <typename T>
void DeviceVector<T>::assign(const DeviceVector<T>& src, chronosity c)
{
	assert(m_capacity >= src.size());

	if /*constexpr*/ (c == chronosity::syn)
		ErrorCheck([&]() { return cudaMemcpy(m_vec, src.m_vec, src.size() * sizeof(T), cudaMemcpyDeviceToDevice); });
	else
		ErrorCheck([&]() { return cudaMemcpyAsync(m_vec, src.data(), src.size() * sizeof(T), cudaMemcpyHostToDevice); });

	m_size = src.size();
}
template <typename T>
void DeviceVector<T>::assign(const std::vector<T>& src)
{
	assert(m_capacity >= src.size());

	ErrorCheck([&]() { return cudaMemcpy(m_vec, src.data(), src.size() * sizeof(T), cudaMemcpyHostToDevice); });

	m_size = src.size();
}

template <typename T>
void DeviceVector<T>::store(const PinnedVector<T>& src, chronosity c, cudaStream_t stream)
{
	if (m_capacity < src.size())
		swap(DeviceVector<T>{ src.size() });
	assign(src, c, stream);
}

template <typename T>
void DeviceVector<T>::store(const DeviceVector<T>& src, chronosity c)
{
	if (m_capacity < src.size())
		swap(DeviceVector<T>{ src.size() });
	assign(src, c);
}

template <typename T>
void DeviceVector<T>::store(const std::vector<T>& src)
{
	if (m_capacity < src.size())
		swap(DeviceVector<T>{ src.size() });
	assign(src);
}

template<typename T>
std::vector<T> DeviceVector<T>::load() const
{
	std::vector<T> ret(m_size);
	ErrorCheck([&]() { return cudaMemcpy(ret.data(), m_vec, m_size * sizeof(T), cudaMemcpyDeviceToHost); });
	return ret;
}

template <typename T>
void DeviceVector<T>::reserve(const std::size_t new_capacity, chronosity c, cudaStream_t stream)
{
	if (new_capacity > m_capacity)
	{
		DeviceVector<T> novum{ new_capacity };
		novum.assign(*this, c, stream);
		swap(novum);
	}
}

template<typename T>
void DeviceVector<T>::swap(DeviceVector<T>& o)
{
	std::swap(m_vec, o.m_vec);
	std::swap(m_size, o.m_size);
	std::swap(m_capacity, o.m_capacity);
}

template<typename T>
void DeviceVector<T>::swap(DeviceVector<T>&& o)
{
	std::swap(m_vec, o.m_vec);
	std::swap(m_size, o.m_size);
	std::swap(m_capacity, o.m_capacity);
}


template <typename T>
__device__ T& DeviceVectorView<T>::at(std::size_t index)
{
	assert(index >= m_size);
	return m_vec[index];
}

template <typename T>
__device__ const T& DeviceVectorView<T>::at(std::size_t index) const
{
	assert(index >= m_size);
	return m_vec[index];
}

template <typename T>
__device__ void DeviceVectorView<T>::push_back(const T& value)
{
	assert(m_size >= m_capacity);
	m_vec[m_size++] = value;
}

template <typename T>
__device__ void DeviceVectorView<T>::push_back(T&& value)
{
	assert(m_size >= m_capacity);
	m_vec[m_size++] = std::move(value);
}

template<typename T>
__device__ void DeviceVectorView<T>::swap(DeviceVectorView<T>& o)
{
	std::swap(m_vec, o.m_vec);
	std::swap(m_size, o.m_size);
	std::swap(m_capacity, o.m_capacity);
}

template<typename T>
__device__ void DeviceVectorView<T>::swap(DeviceVectorView<T>&& o)
{
	std::swap(m_vec, o.m_vec);
	std::swap(m_size, o.m_size);
	std::swap(m_capacity, o.m_capacity);
}
