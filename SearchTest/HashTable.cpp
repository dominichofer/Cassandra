#include "pch.h"

TEST(RAM_HashTable, empty_table_has_no_entries)
{
	RAM_HashTable table{ 1 };
	ASSERT_EQ(table.LookUp(RandomPosition()), std::nullopt);
}

TEST(RAM_HashTable, size_1)
{
	Position pos1 = RandomPosition();
	Position pos2 = PlayPass(pos1);
	RAM_HashTable table{ 1 };

	table.Insert(pos1, Result{ {0, 0}, {1}, Field::A1 });
	ASSERT_NE(table.LookUp(pos1), std::nullopt);
	ASSERT_EQ(table.LookUp(pos2), std::nullopt);
}

TEST(RAM_HashTable, size_1000)
{
	Position pos = RandomPosition();
	RAM_HashTable table{ 1'000 };
	ASSERT_EQ(table.LookUp(pos), std::nullopt);

	table.Insert(pos, Result{ {0, 0}, {1}, Field::A1 });
	ASSERT_NE(table.LookUp(pos), std::nullopt);
	ASSERT_EQ(table.LookUp(PlayPass(pos)), std::nullopt);
}

TEST(FileHashTable, size_1000)
{
	auto tmp = std::filesystem::temp_directory_path();
	auto config = tmp / "config";
	auto tt1 = tmp / "tt1";
	auto tt2 = tmp / "tt2";
	FileHashTable::Create(config, 1'000, { tt1, tt2 });

	FileHashTable table{ config };
	Position pos = RandomPosition();
	EXPECT_EQ(table.LookUp(pos), std::nullopt);

	table.Insert(pos, {});
	EXPECT_NE(table.LookUp(pos), std::nullopt);
	EXPECT_EQ(table.LookUp(PlayPass(pos)), std::nullopt);

	table.Close();
	FileHashTable::Delete(config);
}

TEST(MultiLevelHashTable, size_1000)
{
	Position pos1 = RandomPosition();
	Position pos2 = PlayPass(pos1);
	RAM_HashTable table1{ 1'000 };
	RAM_HashTable table2{ 1'000 };
	MultiLevelHashTable table{ { table1, table2 } };

	table.Insert(pos1, {});
	ASSERT_NE(table1.LookUp(pos1), std::nullopt);
	ASSERT_EQ(table1.LookUp(pos2), std::nullopt);
	ASSERT_NE(table2.LookUp(pos1), std::nullopt);
	ASSERT_EQ(table2.LookUp(pos2), std::nullopt);
	ASSERT_NE(table.LookUp(pos1), std::nullopt);
	ASSERT_EQ(table.LookUp(pos2), std::nullopt);

	table1.Clear();
	ASSERT_EQ(table1.LookUp(pos1), std::nullopt);
	ASSERT_NE(table2.LookUp(pos1), std::nullopt);
	ASSERT_NE(table.LookUp(pos1), std::nullopt);
}