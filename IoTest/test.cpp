#include "pch.h"
#include <chrono>
#include <iostream>
#include <vector>

TEST(SaveLoad, BitBoard)
{
	const BitBoard in = BitBoard::HorizontalLine(3);

	std::stringstream ss;
	Write(ss, in);
	auto out = Read<BitBoard>(ss);

	EXPECT_EQ(in, out);
}

TEST(SaveLoad, Puzzle)
{
	using namespace std::chrono_literals;
	Request request(Field::B3, 13, 3.0_sigmas);
	Result result(+04, 10'000, 3s);
	Puzzle::Task task(request, result);
	const Puzzle in(Position::Start(), { task });

	std::stringstream ss;
	Write(ss, in);
	auto out = Read<Puzzle>(ss);

	EXPECT_EQ(in, out);
}

TEST(SaveLoad, std_vector_int)
{
	std::vector<int> in = { 1,2,3 };

	std::stringstream ss;
	Write(ss, in);
	auto out = Read<std::vector<int>>(ss);

	EXPECT_EQ(in, out);
}

TEST(DataBase, train_test_view)
{
	int train_size = 3;
	int test_size = 2;
	int total_size = 6;
	auto rng = std::default_random_engine{};
	std::vector<std::vector<std::vector<Puzzle>>> input_data;
	for (int f = 0; f < 2; f++)
	{
		std::vector<std::vector<Puzzle>> file;
		for (int e = 0; e <= 60; e++)
		{
			std::set<Position> set = generate_n_unique(total_size, PosGen::RandomWithEmptyCount{ e });
			std::vector<Puzzle> pos(set.begin(), set.end());
			std::shuffle(pos.begin(), pos.end(), rng);
			file.push_back(pos);
		}
		input_data.push_back(file);
	}
	DataBase<Puzzle> data;
	for (const auto& file : input_data)
		data.Add("file", file | std::views::join);

	auto [train_view, test_view] = data | views::train_test(train_size, test_size);

	std::vector<Puzzle> train, test;
	for (const auto& file : input_data)
		for (const auto& of_equal_empty_count : file)
		{
			for (const auto& p : of_equal_empty_count | std::views::take(train_size))
				train.push_back(p);
			for (const auto& p : of_equal_empty_count | std::views::drop(train_size) | std::views::take(test_size))
				test.push_back(p);
		}

	for (const auto& p : train_view)
		EXPECT_TRUE(std::ranges::find(train, p) != train.end());
	for (const auto& p : test_view)
		EXPECT_TRUE(std::ranges::find(test, p) != test.end());
}