#include "pch.h"
#include <chrono>
#include <string>
#include <vector>

TEST(Stream, serialize_deserialize_std_chrono_duration)
{
	using namespace std::chrono_literals;
	const std::chrono::duration<double> in = 3s;

	std::stringstream stream;
	Serialize(in, stream);
	auto out = Deserialize<decltype(in)>(stream);

	EXPECT_EQ(in, out);
}

TEST(Stream, serialize_deserialize_std_string)
{

	const std::string in = "Hello";

	std::stringstream stream;
	Serialize(in, stream);
	auto out = Deserialize<decltype(in)>(stream);

	EXPECT_EQ(in, out);
}

TEST(Stream, serialize_deserialize_std_vector_int)
{
	const std::vector<int> in = { 1,2,3 };

	std::stringstream stream;
	Serialize(in, stream);
	auto out = Deserialize<decltype(in)>(stream);

	EXPECT_EQ(in, out);
}