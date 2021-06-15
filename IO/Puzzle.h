#pragma once
#include "Core/Core.h"
#include "Search/Search.h"
#include "File.h"
#include <filesystem>
#include <vector>

//class DataBase
//{
//	std::vector<Puzzle> vec;
//public:
//	DataBase() noexcept = default;
//	DataBase(const std::vector<std::filesystem::path>& projects);
//
//	void Add(const std::filesystem::path& proj) {
//		PuzzleProject proj = LoadProject<Puzzle>(file);
//		vec.push_back(Load); }
//};

//#pragma pack(1)
//struct CompactPuzzle
//{
//	Position pos{};
//	uint64 nodes = 0;
//	std::chrono::duration<double> duration{0};
//	std::vector<Search::Request> request;
//	std::vector<Search::Result> result;
//
//	CompactPuzzle() noexcept = default;
//	CompactPuzzle(const Puzzle& p) noexcept : pos(p.Position()), nodes(p.Nodes()), duration(p.Duration()), request(p.Request()), result(p.Result()) {}
//	operator Puzzle() const { return { pos, nodes, duration, request, result }; }
//};
//#pragma pack()
//
//template <>
//struct compact<Puzzle>
//{
//	using type = typename CompactPuzzle;
//};
//
//
//template <typename value_type>
//[[nodiscard]]
//std::vector<value_type> Load(const std::filesystem::path& file)
//{
//	std::fstream stream(file, std::ios::in | std::ios::binary);
//	if (!stream.is_open())
//		throw std::fstream::failure("Can not open '" + file.string() + "' for binary intput.");
//
//	stream.seekg(0, stream.end);
//	const int size = stream.tellg() / sizeof(dense_t<value_type>);
//	stream.seekg(0, stream.beg);
//
//	std::vector<value_type> data;
//	data.reserve(size);
//	dense_t<value_type> buffer;
//	for (int i = 0; i < size; i++)
//	{
//		stream.read(reinterpret_cast<char*>(&buffer), sizeof buffer);
//		data.push_back(buffer);
//	}
//	return data;
//}
//
//template <typename Iterator>
//void Save(const std::filesystem::path& file, Iterator begin, Iterator end)
//{
//	std::fstream stream(file, std::ios::out | std::ios::binary);
//	if (!stream.is_open())
//		throw std::fstream::failure("Can not open '" + file.string() + "' for binary output.");
//
//	for (auto it = begin; it < end; ++it)
//	{
//		dense_t<Iterator::value_type> buffer{*it};
//		stream.write(reinterpret_cast<const char*>(std::addressof(buffer)), sizeof buffer);
//	}
//}