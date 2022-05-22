#include "pch.h"
#include <algorithm>
#include <string>
#include <vector>

class DataBaseFixture : public ::testing::Test
{
public:
	DB<int> db;

	DataBaseFixture()
	{
		using namespace std::string_literals;
		db.Add({}, 0);
		db.Add({ "A" }, 1);
		db.Add({ "B" }, 2);
		db.Add({ "A", "B" }, std::vector{ 3, 4 });
		db.Add({ "A", "B", "C" }, 5);
	}
};

TEST_F(DataBaseFixture, plan_access)
{
	EXPECT_TRUE(ranges::equal(
		db,
		std::vector{ 0,1,2,3,4,5 }
	));
}

TEST_F(DataBaseFixture, where_meta)
{
	EXPECT_TRUE(ranges::equal(
		db.Where("A"),
		std::vector{ 1,3,4,5 }
	));
}

TEST_F(DataBaseFixture, where_data_predicate)
{
	EXPECT_TRUE(ranges::equal(
		db.Where([](const int& data) { return data % 2 == 0; }),
		std::vector{ 0,2,4 }
	));
}

TEST_F(DataBaseFixture, where_any_of_meta)
{
	EXPECT_TRUE(ranges::equal(
		db.WhereAnyOf({ "A", "B" }),
		std::vector{ 1,2,3,4,5 }
	));
}

TEST_F(DataBaseFixture, where_all_of_meta)
{
	EXPECT_TRUE(ranges::equal(
		db.WhereAllOf({ "A", "B" }),
		std::vector{ 3,4,5 }
	));
}

TEST_F(DataBaseFixture, where_none_of_meta)
{
	EXPECT_TRUE(ranges::equal(
		db.WhereNoneOf({ "A", "B" }),
		std::vector{ 0 }
	));
}

TEST_F(DataBaseFixture, serialize_deserialize)
{
	std::stringstream stream;
	Serialize(db, stream);
	auto db2 = Deserialize<decltype(db)>(stream);
	EXPECT_TRUE(ranges::equal(db, db2));
}
