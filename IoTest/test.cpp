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