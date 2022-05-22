#pragma once
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#include <valarray>
#include <istream>
#include <ostream>

template<class, template<class...> class>
inline constexpr bool is_specialization = false;

template<template<class...> class T, class... Args>
inline constexpr bool is_specialization<T<Args...>, T> = true;


template <typename T>
void Serialize(const T& t, std::ostream& stream) requires std::is_arithmetic_v<T> or std::is_enum_v<T>
{
	stream.write(reinterpret_cast<const char*>(std::addressof(t)), sizeof t);
}

template <typename T>
T Deserialize(std::istream& stream) requires std::is_arithmetic_v<T> or std::is_enum_v<T>
{
	T t;
	stream.read(reinterpret_cast<char*>(std::addressof(t)), sizeof t);
	return t;
}


template <typename T>
T Deserialize(std::istream& stream) requires std::is_const_v<T>
{
	return Deserialize<std::remove_const_t<T>>(stream);
}


template <typename T>
void Serialize(const T& t, const std::filesystem::path& file)
{
	std::fstream stream(file, std::ios::binary | std::ios::out);
	Serialize(t, stream);
}

template <typename T>
T Deserialize(const std::filesystem::path& file)
{
	std::fstream stream(file, std::ios::binary | std::ios::in);
	return Deserialize<T>(stream);
}

template <typename T>
concept HasSerialize = requires (const T& t, std::ostream & stream) { t.Serialize(stream); };

void Serialize(const HasSerialize auto& t, std::ostream& stream)
{
	// Converts member function to free function.
	t.Serialize(stream);
}

template <typename T>
T Deserialize(std::istream& stream) requires requires { T::Deserialize(stream); }
{
	// Converts member function to free function.
	return T::Deserialize(stream);
}


template <std::ranges::range T>
void Serialize(const T& rng, std::ostream& stream)
requires (not HasSerialize<T>)
{
	static_assert(not HasSerialize<T>);
	Serialize(std::size(rng), stream);
	for (const auto& r : rng)
		Serialize(r, stream);
}

void Serialize(auto begin, auto end, std::ostream& stream)
{
	Serialize(std::ranges::subrange(begin, end), stream);
}

template <typename T>
T Deserialize(std::istream& stream) requires is_specialization<T, std::vector>
{
	auto size = Deserialize<std::size_t>(stream);
	T vec;
	vec.reserve(size);
	for (std::size_t i = 0; i < size; i++)
		vec.push_back(Deserialize<typename T::value_type>(stream));
	return vec;
}

template <typename T>
T Deserialize(std::istream& stream) requires is_specialization<T, std::valarray>
{
	auto size = Deserialize<std::size_t>(stream);
	T vec(size);
	for (std::size_t i = 0; i < size; i++)
		vec[i] = Deserialize<typename T::value_type>(stream);
	return vec;
}


// std::chrono::duration
template <typename T>
void Serialize(T duration, std::ostream& stream) requires is_specialization<T, std::chrono::duration>
{
	Serialize(duration.count(), stream);
}
template <typename T>
T Deserialize(std::istream& stream) requires is_specialization<T, std::chrono::duration>
{
	return T{ Deserialize<T::rep>(stream) };
}

// std::string
void Serialize(const std::string&, std::ostream&);
template <>
std::string Deserialize<std::string>(std::istream&);


// TODO: Remove!
template <typename T>
std::vector<T> LoadVec_old(const std::filesystem::path& file)
{
	std::ifstream stream(file, std::ios::binary);
	if (!stream.is_open())
		throw std::istream::failure("Can not open '" + file.string() + "' for binary intput.");

	stream.seekg(0, stream.end);
	const std::size_t size = stream.tellg() / sizeof(T);
	stream.seekg(0, stream.beg);
	
	std::vector<T> data;
	data.reserve(size);
	T buffer;
	for (std::size_t i = 0; i < size; i++)
	{
		stream.read(reinterpret_cast<char*>(&buffer), sizeof buffer);
		data.push_back(buffer);
	}
	return data;
}

// TODO: Remove!
template <typename T>
void Save_old(const std::filesystem::path& file, const T& vec)
{
	std::ofstream stream(file, std::ios::binary);
	if (!stream.is_open())
		throw std::ostream::failure("Can not open '" + file.string() + "' for binary intput.");

	for (const auto& x : vec)
		stream.write(reinterpret_cast<const char*>(std::addressof(x)), sizeof T::value_type);
}

//template <typename T>
//struct compact
//{
//	using type = typename T;
//};
//
//template <typename T>
//using dense_t = typename compact<T>::type;
//
//
//template <typename Iterator>
//void Save(std::ostream& stream, const Iterator& begin, const Iterator& end)
//{
//	for (auto it = begin; it < end; ++it)
//	{
//		dense_t<Iterator::value_type> buffer{ *it };
//		stream.write(reinterpret_cast<const char*>(std::addressof(buffer)), sizeof buffer);
//	}
//}
//
//template <typename Iterator>
//void Save(const std::filesystem::path& file, const Iterator& begin, const Iterator& end)
//{
//	std::ostream stream(file, std::ios::binary);
//	if (!stream.is_open())
//		throw std::ostream::failure("Can not open '" + file.string() + "' for binary output.");
//
//	Save(stream, begin, end);
//}
//
//template <typename Container>
//void Save(std::ostream& stream, const Container& c)
//{
//	Save(stream, c.begin(), c.end());
//}
//
//template <typename Container>
//void Save(const std::filesystem::path& file, const Container& c)
//{
//	Save(file, c.begin(), c.end());
//}
//
//template <typename value_type>
//std::vector<value_type> Load_dense(std::istream& stream)
//{
//	stream.seekg(0, stream.end);
//	const std::size_t size = stream.tellg() / sizeof(dense_t<value_type>);
//	stream.seekg(0, stream.beg);
//
//	std::vector<value_type> data;
//	data.reserve(size);
//	dense_t<value_type> buffer;
//	for (std::size_t i = 0; i < size; i++)
//	{
//		stream.read(reinterpret_cast<char*>(&buffer), sizeof buffer);
//		data.push_back(buffer);
//	}
//	return data;
//}
//
//template <typename value_type>
//std::vector<value_type> Load(const std::filesystem::path& file)
//{
//	std::istream stream(file, std::ios::binary);
//	if (!stream.is_open())
//		throw std::istream::failure("Can not open '" + file.string() + "' for binary intput.");
//
//	return Load<value_type>(stream);
//}

//template <typename Stream = std::fstream>
//class BinaryFileStream
//{
//	Stream stream;
//public:
//	BinaryFileStream() = default;
//	BinaryFileStream(const std::filesystem::path& file) : stream(file, std::ios::binary | std::ios::in | std::ios::out)
//	{
//		if (not stream.is_open())
//			throw std::fstream::failure("Can not open '" + file.string() + "' for binary intput/output.");
//	}
//
//	void close() { stream.close(); }
//
//	template <typename T>
//	T read() requires std::is_arithmetic_v<T> or std::is_enum_v<T>
//	{
//		T t;
//		stream.read(reinterpret_cast<char*>(std::addressof(t)), sizeof t);
//		return t;
//	}
//
//	template <typename T>
//	void write(const T& t) requires std::is_arithmetic_v<T> or std::is_enum_v<T>
//	{
//		stream.write(reinterpret_cast<const char*>(std::addressof(t)), sizeof t);
//	}
//};
//
//// deduction guide
//BinaryFileStream(const std::filesystem::path& file)->BinaryFileStream<std::fstream>;