#include "benchmark/benchmark.h"
#include "Search/Search.h"

void Hash(benchmark::State& state)
{
	Position pos = RandomPosition();
	for (auto _ : state)
	{
		auto ret = ::Hash(pos);
		benchmark::DoNotOptimize(ret);
	}
}
BENCHMARK(Hash);

void RAM_HashTable_Insert(benchmark::State& state)
{
	RAM_HashTable table{ 1'000'000 };

	Position pos = RandomPosition();
	for (auto _ : state)
	{
		table.Insert(pos, {});
	}
}
BENCHMARK(RAM_HashTable_Insert);

void RAM_HashTable_LookUp(benchmark::State& state)
{
	RAM_HashTable table{ 1'000'000 };

	Position pos = RandomPosition();
	for (auto _ : state)
	{
		auto ret = table.LookUp(pos);
		benchmark::DoNotOptimize(ret);
	}
}
BENCHMARK(RAM_HashTable_LookUp);

void RAM_HashTable_LookUp2(benchmark::State& state)
{
	RAM_HashTable table{ 1'000'000 };

	Position pos = RandomPosition();
	table.Insert(pos, {});
	for (auto _ : state)
	{
		auto ret = table.LookUp(pos);
		benchmark::DoNotOptimize(ret);
	}
}
BENCHMARK(RAM_HashTable_LookUp2);

std::filesystem::path CreateFileHashTable()
{
	auto tmp = std::filesystem::temp_directory_path();
	auto config = tmp / "config";
	auto tt1 = tmp / "tt1";
	auto tt2 = tmp / "tt2";
	FileHashTable::Create(config, 1'000'000, { tt1, tt2 });
	return config;
}

void FileHashTable_Insert(benchmark::State& state)
{
	auto config = CreateFileHashTable();
	FileHashTable table{ config };
	Position pos = RandomPosition();
	for (auto _ : state)
		table.Insert(pos, {});

	table.Close();
	FileHashTable::Delete(config);
}
BENCHMARK(FileHashTable_Insert);

void FileHashTable_LookUp(benchmark::State& state)
{
	auto config = CreateFileHashTable();
	FileHashTable table{ config };
	Position pos = RandomPosition();
	for (auto _ : state)
		benchmark::DoNotOptimize(table.LookUp(pos));

	table.Close();
	FileHashTable::Delete(config);
}
BENCHMARK(FileHashTable_LookUp);

void MultiLevelHashTable_Insert(benchmark::State& state)
{
	RAM_HashTable table1{ 1'000'000 };
	RAM_HashTable table2{ 1'000'000 };
	MultiLevelHashTable table{ { table1, table2 } };
	Position pos = RandomPosition();
	for (auto _ : state)
		table.Insert(pos, {});
}
BENCHMARK(MultiLevelHashTable_Insert);

void MultiLevelHashTable_LookUp(benchmark::State& state)
{
	RAM_HashTable table1{ 1'000'000 };
	RAM_HashTable table2{ 1'000'000 };
	MultiLevelHashTable table{ { table1, table2 } };
	Position pos = RandomPosition();
	for (auto _ : state)
		benchmark::DoNotOptimize(table.LookUp(pos));
}
BENCHMARK(MultiLevelHashTable_LookUp);

void MultiLevelHashTable2_Insert(benchmark::State& state)
{
	RAM_HashTable table1{ 1'000'000 };
	auto config = CreateFileHashTable();
	FileHashTable table2{ config };
	MultiLevelHashTable table{ { table1, table2 } };
	Position pos = RandomPosition();
	for (auto _ : state)
		table.Insert(pos, {});

	table2.Close();
	FileHashTable::Delete(config);
}
BENCHMARK(MultiLevelHashTable2_Insert);

void MultiLevelHashTable2_LookUp(benchmark::State& state)
{
	RAM_HashTable table1{ 1'000'000 };
	auto config = CreateFileHashTable();
	FileHashTable table2{ config };
	MultiLevelHashTable table{ { table1, table2 } };
	Position pos = RandomPosition();
	for (auto _ : state)
		benchmark::DoNotOptimize(table.LookUp(pos));

	table2.Close();
	FileHashTable::Delete(config);
}
BENCHMARK(MultiLevelHashTable2_LookUp);