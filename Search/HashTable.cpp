#include "HashTable.h"

uint64_t Hash(const Position& pos)
{
	const uint64_t kMul = 0x9ddfea08eb382d69ULL;
	uint64_t a = pos.Player() * kMul;
	a ^= (a >> 47);
	uint64_t b = (pos.Opponent() ^ a) * kMul;
	b ^= (b >> 47);
	return b;
}

void Bucket::Insert(const Position& new_pos, const Result& new_result)
{
	pos = new_pos;
	result = new_result;
}

std::optional<Result> Bucket::LookUp(const Position& pos) const
{
	if (pos == this->pos)
		return result;
	return std::nullopt;
}

void Bucket::Clear()
{
	pos = { 0, 0 };
	result = {};
}

void RAM_HashTable::Insert(const Position& pos, const Result& result)
{
	uint64_t index = Hash(pos);
	std::unique_lock lock(mutexes[index & 0xFFULL]);
	buckets[index % buckets.size()].Insert(pos, result);
}

std::optional<Result> RAM_HashTable::LookUp(const Position& pos) const
{
	uint64_t index = Hash(pos);
	std::shared_lock lock(mutexes[index & 0xFFULL]);
	return buckets[index % buckets.size()].LookUp(pos);
}

void RAM_HashTable::Clear()
{
	for (Bucket& b : buckets)
		b.Clear();
}

FileHashTable::FileHashTable(std::filesystem::path config)
{
	std::ifstream stream(config);

	// Read the first entry as a number
	stream >> file_size;

	// Read the remaining entries as file paths
	std::filesystem::path path;
	while (stream >> path)
		streams.emplace_back(path, std::ios::binary | std::ios::in | std::ios::out);
}

void FileHashTable::Create(std::filesystem::path config, std::size_t file_size, const std::vector<std::filesystem::path> files)
{
	std::ofstream config_stream(config);

	// Write the first entry as a number
	config_stream << file_size << '\n';

	// Write the remaining entries as file paths
	for (const std::filesystem::path& path : files)
		config_stream << path << '\n';

	// Create the first file
	Bucket bucket{};
	std::fstream stream(files.front(), std::ios::binary | std::ios::out);
	for (size_t i = 0; i < file_size; ++i)
		stream.write(reinterpret_cast<const char*>(std::addressof(bucket)), sizeof(Bucket));
	stream.close();

	// Copy the first file to the remaining files
	for (size_t i = 1; i < files.size(); ++i)
		std::filesystem::copy(files.front(), files[i]);
}

void FileHashTable::Delete(std::filesystem::path config)
{
	std::ifstream stream(config);

	// Read the first entry as a number
	std::size_t file_size;
	stream >> file_size;

	// Read the remaining entries as file paths
	std::filesystem::path path;
	while (stream >> path)
		std::filesystem::remove(path);

	stream.close();

	// Delete the config file
	std::filesystem::remove(config);
}

void FileHashTable::Insert(const Position& pos, const Result& result)
{
	uint64_t index = Hash(pos);
	std::unique_lock lock(mutexes[index & 0xFFULL]);

	Bucket bucket;
	std::fstream& stream = streams[(index / file_size) % streams.size()];
	stream.seekg((index % file_size) * sizeof(Bucket), std::ios::beg);
	stream.read(reinterpret_cast<char*>(std::addressof(bucket)), sizeof(Bucket));
	if (stream.fail())
		throw std::runtime_error("Failed to read from file");
	bucket.Insert(pos, result);
	stream.seekp((index % file_size) * sizeof(Bucket), std::ios::beg);
	stream.write(reinterpret_cast<const char*>(std::addressof(bucket)), sizeof(Bucket));
}

std::optional<Result> FileHashTable::LookUp(const Position& pos) const
{
	uint64_t index = Hash(pos);
	std::shared_lock lock(mutexes[index & 0xFFULL]);

	Bucket bucket{ Position{}, Result{} };
	std::fstream& stream = streams[(index / file_size) % streams.size()];
	stream.seekp((index % file_size) * sizeof(Bucket), std::ios::beg);
	stream.read(reinterpret_cast<char*>(std::addressof(bucket)), sizeof(Bucket));
	if (stream.fail())
		throw std::runtime_error("Failed to read from file");
	return bucket.LookUp(pos);
}

void FileHashTable::Clear()
{
	Bucket bucket{};
	for (std::fstream& stream : streams)
	{
		stream.seekg(0, std::ios::beg);
		for (size_t i = 0; i < file_size; ++i)
			stream.write(reinterpret_cast<const char*>(std::addressof(bucket)), sizeof(Bucket));
	}
}

void FileHashTable::Close()
{
	for (auto& s : streams)
		s.close();
}

void MultiLevelHashTable::Insert(const Position& pos, const Result& result)
{
	for (HashTable& ht : levels)
		ht.Insert(pos, result);
}

std::optional<Result> MultiLevelHashTable::LookUp(const Position& pos) const
{
	for (HashTable& ht : levels)
		if (auto result = ht.LookUp(pos))
			return result;
	return std::nullopt;
}

void MultiLevelHashTable::Clear()
{
	for (HashTable& ht : levels)
		ht.Clear();
}
