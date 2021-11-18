#pragma once
#include "Search/Search.h"
#include <vector>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <shared_mutex>

template<class, template<class...> class>
inline constexpr bool is_specialization = false;
template<template<class...> class T, class... Args>
inline constexpr bool is_specialization<T<Args...>, T> = true;

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
//[[nodiscard]]
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
//[[nodiscard]]
//std::vector<value_type> Load(const std::filesystem::path& file)
//{
//	std::istream stream(file, std::ios::binary);
//	if (!stream.is_open())
//		throw std::istream::failure("Can not open '" + file.string() + "' for binary intput.");
//
//	return Load<value_type>(stream);
//}

template <typename Stream = std::fstream>
class BinaryFileStream
{
	Stream stream;
public:
	BinaryFileStream() = default;
	BinaryFileStream(const std::filesystem::path& file) : stream(file, std::ios::binary)
	{
		if (not stream.is_open())
			throw std::fstream::failure("Can not open '" + file.string() + "' for binary intput/output.");
	}

	void close() { stream.close(); }

	//template <typename T>
	//void write(const T& t);

	//template <typename T>
	//T read();

	template <typename T>
	requires std::is_arithmetic_v<T> or std::is_enum_v<T>
	void write(const T& t)
	{
		stream.write(reinterpret_cast<const char*>(std::addressof(t)), sizeof(T));
	}

	template <typename T>
	requires std::is_arithmetic_v<T> or std::is_enum_v<T>
	T read()
	{
		T t;
		stream.read(reinterpret_cast<char*>(std::addressof(t)), sizeof(T));
		return t;
	}

	void write(const std::ranges::range auto& range)
	{
		write(std::size(range));
		for (const auto& elem : range)
			write(elem);
	}

	void write(auto begin, auto end)
	{
		write(std::ranges::subrange(begin, end));
	}

	template <typename T>
	requires is_specialization<T, std::vector>
	T read()
	{
		auto size = read<std::size_t>();
		T vec;
		vec.reserve(size);
		for (std::size_t i = 0; i < size; i++)
			vec.push_back(read<typename T::value_type>());
		return vec;
	}

	void write(const BitBoard& obj)
	{
		write(static_cast<uint64_t>(obj));
	}

	template <typename T>
	requires std::is_same_v<T, BitBoard>
	T read()
	{
		return read<uint64_t>();
	}

	void write(const Position& pos)
	{
		write(pos.Player());
		write(pos.Opponent());
	}

	template <typename T>
	requires std::is_same_v<T, Position>
	T read()
	{
		auto P = read<uint64_t>();
		auto O = read<uint64_t>();
		return { P, O };
	}

	void write(const Confidence& selectivity)
	{
		write(selectivity.sigmas());
	}

	template <typename T>
	requires std::is_same_v<T, Confidence>
	T read()
	{
		return Confidence{ read<decltype(std::declval<Confidence>().sigmas())>() };
	}

	void write(const Intensity& i)
	{
		write(i.depth);
		write(i.certainty);
	}

	template <typename T>
	requires std::is_same_v<T, Intensity>
	T read()
	{
		auto depth = read<decltype(Intensity::depth)>();
		auto certainty = read<decltype(Intensity::certainty)>();
		return { depth, certainty };
	}

	void write(const std::chrono::duration<double>& duration)
	{
		write(duration.count());
	}

	template <typename T>
	requires std::is_same_v<T, std::chrono::duration<double>>
	T read()
	{
		return std::chrono::duration<double>(read<double>());
	}

	void write(const Request& r)
	{
		write(r.move);
		write(r.intensity);
	}
	
	template <typename T>
	requires std::is_same_v<T, Request>
	T read()
	{
		auto move = read<decltype(Request::move)>();
		auto intensity = read<decltype(Request::intensity)>();
		return { move, intensity };
	}

	void write(const Result& r)
	{
		write(r.score);
		write(r.nodes);
		write(r.duration);
	}
	
	template <typename T>
	requires std::is_same_v<T, Result>
	T read()
	{
		auto score = read<decltype(Result::score)>();
		auto nodes = read<decltype(Result::nodes)>();
		auto duration = read<decltype(Result::duration)>();
		return { score, nodes, duration };
	}

	void write(const Puzzle::Task& task)
	{
		write(task.GetRequest());
		write(task.GetResult());
	}
	
	template <typename T>
	requires std::is_same_v<T, Puzzle::Task>
	T read()
	{
		auto request = read<Request>();
		auto result = read<Result>();
		return T(request, result);
	}

	void write(const Puzzle& p)
	{
		write(p.pos);
		write(p.tasks);
	}
	
	template <typename T>
	requires std::is_same_v<T, Puzzle>
	T read()
	{
		auto pos = read<decltype(Puzzle::pos)>();
		auto tasks = read<decltype(Puzzle::tasks)>();

		return { pos, std::move(tasks) };
	}
};

// deduction guide
BinaryFileStream(const std::filesystem::path& file) -> BinaryFileStream<std::fstream>;


//// Project
//template <typename T>
//void Write(std::ostream& stream, const Project<T>& proj)
//{
//	std::unique_lock lock(proj.mutex);
//	Write(stream, proj.size());
//	for (const T& wu : proj)
//		Write(stream, wu);
//}
//
//// Project
//template <typename T, std::enable_if_t<std::is_same_v<T, Project<typename T::value_type>>, bool> = true>
//[[nodiscard]]
//T Read(std::istream& stream)
//{
//	std::size_t size = Read<std::size_t>(stream);
//	T proj;
//	proj.reserve(size);
//	for (std::size_t i = 0; i < size; i++)
//		proj.push_back(Read<T::value_type>(stream));
//	return proj;
//}
//
//// PuzzleProject
//template <typename T, std::enable_if_t<std::is_same_v<T, PuzzleProject>, bool> = true>
//[[nodiscard]]
//T Read(std::istream& stream)
//{
//	std::size_t size = Read<std::size_t>(stream);
//	T proj;
//	proj.reserve(size);
//	for (std::size_t i = 0; i < size; i++)
//		proj.push_back(Read<T::value_type>(stream));
//	return proj;
//}

template <typename T>
void Save(const std::filesystem::path& file, const T& t)
{
	BinaryFileStream{ file }.write(t);
}

template <typename Iterator>
void Save(const std::filesystem::path& file, Iterator first, Iterator last)
{
	BinaryFileStream{ file }.write(first, last);
}

template <typename T>
[[nodiscard]]
T Load(const std::filesystem::path& file)
{
	return BinaryFileStream{ file }.read<T>();
}

template <typename T>
[[nodiscard]]
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

template <typename T>
[[nodiscard]]
void Save_old(const std::filesystem::path& file, const T& vec)
{
	std::ofstream stream(file, std::ios::binary);
	if (!stream.is_open())
		throw std::ostream::failure("Can not open '" + file.string() + "' for binary intput.");

	for (const auto& x : vec)
		stream.write(reinterpret_cast<const char*>(std::addressof(x)), sizeof T::value_type);
}