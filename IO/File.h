#pragma once
#include "Search/Search.h"
#include <vector>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <shared_mutex>

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



// Arithmetic or Enum
template <typename T, std::enable_if_t<std::is_arithmetic_v<T> or std::is_enum_v<T>, bool> = true>
void Write(std::ostream& stream, const T& t)
{
	stream.write(reinterpret_cast<const char*>(std::addressof(t)), sizeof(T));
}

// Arithmetic or Enum
template <typename T, std::enable_if_t<std::is_arithmetic_v<T> or std::is_enum_v<T>, bool> = true>
[[nodiscard]]
T Read(std::istream& stream)
{
	T t;
	stream.read(reinterpret_cast<char*>(std::addressof(t)), sizeof(T));
	return t;
}

template <typename T, typename = void>
struct is_iterable : std::false_type {};

template <typename T>
struct is_iterable<T, std::void_t<decltype(std::begin(std::declval<T>())), decltype(std::end(std::declval<T>()))>> : std::true_type {};

template <typename T>
constexpr bool is_iterable_v = is_iterable<T>::value;

// Iterable
template <typename T, std::enable_if_t<is_iterable_v<T>, bool> = true>
void Write(std::ostream& stream, const T& iterable)
{
	auto begin = std::begin(iterable);
	auto end = std::end(iterable);
	auto distance = std::distance(begin, end);
	Write<std::size_t>(stream, distance);
	for (const auto& t : iterable)
		Write(stream, t);
}

// Iterator
template <typename Iterator>
void Write(std::ostream& stream, Iterator first, Iterator last)
{
	auto distance = std::distance(first, last);
	Write<std::size_t>(stream, distance);
	for (; first != last; ++first)
		Write(stream, *first);
}

// std::vector
template <typename T>
void Write(std::ostream& stream, const std::vector<T>& vec)
{
	Write<std::size_t>(stream, vec.size());
	for (const T& t : vec)
		Write(stream, t);
}

// std::vector
template <typename T, std::enable_if_t<std::is_same_v<T, std::vector<typename T::value_type, typename T::allocator_type>>, bool> = true>
[[nodiscard]]
T Read(std::istream& stream)
{
	std::size_t size = Read<std::size_t>(stream);
	T vec;
	vec.reserve(size);
	for (std::size_t i = 0; i < size; i++)
		vec.push_back(Read<T::value_type>(stream));
	return vec;
}

// BitBoard
inline void Write(std::ostream& stream, const BitBoard& b)
{
	Write(stream, static_cast<uint64_t>(b));
}

// BitBoard
template <typename T, std::enable_if_t<std::is_same_v<T, BitBoard>, bool> = true>
[[nodiscard]]
BitBoard Read(std::istream& stream)
{
	return BitBoard(Read<uint64_t>(stream));
}

// Position
inline void Write(std::ostream& stream, const Position& pos)
{
	Write(stream, pos.Player());
	Write(stream, pos.Opponent());
}

// Position
template <typename T, std::enable_if_t<std::is_same_v<T, Position>, bool> = true>
[[nodiscard]]
Position Read(std::istream& stream)
{
	auto P = Read<uint64_t>(stream);
	auto O = Read<uint64_t>(stream);
	return Position(P, O);
}

// Confidence
inline void Write(std::ostream& stream, const Confidence& selectivity)
{
	Write(stream, selectivity.sigmas());
}

// Confidence
template <typename T, std::enable_if_t<std::is_same_v<T, Confidence>, bool> = true>
[[nodiscard]]
Confidence Read(std::istream& stream)
{
	return Confidence(Read<decltype(std::declval<Confidence>().sigmas())>(stream));
}

// Search::Intensity
inline void Write(std::ostream& stream, const Search::Intensity& intensity)
{
	Write(stream, intensity.depth);
	Write(stream, intensity.certainty);
}

// Search::Intensity
template <typename T, std::enable_if_t<std::is_same_v<T, Search::Intensity>, bool> = true>
[[nodiscard]]
Search::Intensity Read(std::istream& stream)
{
	auto depth = Read<decltype(Search::Intensity::depth)>(stream);
	auto certainty = Read<decltype(Search::Intensity::certainty)>(stream);
	return Search::Intensity(depth, certainty);
}

// std::chrono::duration<double>
inline void Write(std::ostream& stream, const std::chrono::duration<double>& duration)
{
	Write(stream, duration.count());
}

// std::chrono::duration<double>
template <typename T, std::enable_if_t<std::is_same_v<T, std::chrono::duration<double>>, bool> = true>
[[nodiscard]]
std::chrono::duration<double> Read(std::istream& stream)
{
	return std::chrono::duration<double>(Read<double>(stream));
}

// Request
inline void Write(std::ostream& stream, const Request& request)
{
	Write(stream, request.move);
	Write(stream, request.intensity);
}

// Request
template <typename T, std::enable_if_t<std::is_same_v<T, Request>, bool> = true>
[[nodiscard]]
Request Read(std::istream& stream)
{
	auto move = Read<decltype(Request::move)>(stream);
	auto intensity = Read<decltype(Request::intensity)>(stream);
	return Request(move, intensity);
}

// Result
template <typename T, std::enable_if_t<std::is_same_v<T, Result>, bool> = true>
[[nodiscard]]
Result Read(std::istream& stream)
{
	auto score = Read<decltype(Result::score)>(stream);
	auto nodes = Read<decltype(Result::nodes)>(stream);
	auto duration = Read<decltype(Result::duration)>(stream);
	return Result(score, nodes, duration);
}

// Result
inline void Write(std::ostream& stream, const Result& result)
{
	Write(stream, result.score);
	Write(stream, result.nodes);
	Write(stream, result.duration);
}

// Puzzle::Task
inline void Write(std::ostream& stream, const Puzzle::Task& task)
{
	Write(stream, task.request);
	Write(stream, task.result);
}

// Puzzle::Task
template <typename T, std::enable_if_t<std::is_same_v<T, Puzzle::Task>, bool> = true>
[[nodiscard]]
Puzzle::Task Read(std::istream& stream)
{
	auto request = Read<decltype(Puzzle::Task::request)>(stream);
	auto result = Read<decltype(Puzzle::Task::result)>(stream);
	return Puzzle::Task(request, result);
}

// Puzzle
inline void Write(std::ostream& stream, const Puzzle& puzzle)
{
	Write(stream, puzzle.pos);
	Write(stream, puzzle.tasks);
}

// Puzzle
template <typename T, std::enable_if_t<std::is_same_v<T, Puzzle>, bool> = true>
[[nodiscard]]
Puzzle Read(std::istream& stream)
{
	auto pos = Read<decltype(Puzzle::pos)>(stream);
	auto tasks = Read<decltype(Puzzle::tasks)>(stream);

	return Puzzle(pos, std::move(tasks));
}

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
	std::ofstream stream(file, std::ios::binary);
	if (!stream.is_open())
		throw std::ostream::failure("Can not open '" + file.string() + "' for binary output.");
	Write(stream, t);
}

template <typename Iterator>
void Save(const std::filesystem::path& file, Iterator first, Iterator last)
{
	std::ofstream stream(file, std::ios::binary);
	if (!stream.is_open())
		throw std::ostream::failure("Can not open '" + file.string() + "' for binary output.");
	Write(stream, first, last);
}

template <typename T>
[[nodiscard]]
T Load(const std::filesystem::path& file)
{
	std::ifstream stream(file, std::ios::binary);
	if (!stream.is_open())
		throw std::istream::failure("Can not open '" + file.string() + "' for binary intput.");
	return Read<T>(stream);
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