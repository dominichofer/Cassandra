#pragma once
#include "Game/Game.h"
#include "Result.h"
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <optional>
#include <shared_mutex>
#include <vector>

uint64_t Hash(const Position&);

// Interface
struct HashTable
{
	virtual void Insert(const Position&, const Result&) = 0;
	virtual std::optional<Result> LookUp(const Position&) const = 0;
	virtual void Clear() = 0;
};

struct Bucket
{
	Position pos{ 0, 0 };
	Result result{};

	void Insert(const Position&, const Result&);
	std::optional<Result> LookUp(const Position&) const;
	void Clear();
};

class RAM_HashTable final : public HashTable
{
	mutable std::array<std::shared_mutex, 256> mutexes;
	std::vector<Bucket> buckets;
public:
	RAM_HashTable(std::size_t size) : buckets(size) {}

	void Insert(const Position&, const Result&) override;
	std::optional<Result> LookUp(const Position&) const override;
	void Clear() override;
};

class FileHashTable final : public HashTable
{
	std::size_t file_size;
	mutable std::vector<std::fstream> streams;
	mutable std::array<std::shared_mutex, 256> mutexes;
public:
	FileHashTable(std::filesystem::path config);
	static void Create(std::filesystem::path config, std::size_t file_size, const std::vector<std::filesystem::path> files);
	static void Delete(std::filesystem::path config);

	void Insert(const Position&, const Result&) override;
	std::optional<Result> LookUp(const Position&) const override;
	void Clear() override;
	void Close();
};

class MultiLevelHashTable final : public HashTable
{
	std::vector<std::reference_wrapper<HashTable>> levels;
public:
	MultiLevelHashTable(std::vector<std::reference_wrapper<HashTable>> levels) : levels(levels) {}

	void Insert(const Position&, const Result&) override;
	std::optional<Result> LookUp(const Position&) const override;
	void Clear() override;
};
