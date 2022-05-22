#include "pch.h"
#include "CoreIO/CoreIO.h"
#include "Game/Game.h"
#include "GameIO/GameIO.h"
#include <chrono>

using namespace std::chrono_literals;

TEST(Stream, serialize_deserialize_NoMovePuzzle)
{
	Intensity request(13, 3.0_sigmas);
	int result = +04;
	NoMovePuzzle::Task task(request, result);
	const NoMovePuzzle in(Position::Start(), { task });

	std::stringstream stream;
	Serialize(in, stream);
	auto out = Deserialize<decltype(in)>(stream);

	EXPECT_EQ(in, out);
}

TEST(Stream, serialize_deserialize_AllMovePuzzle)
{
	const AllMovePuzzle in(Position::Start(), {13, 3.0_sigmas });

	std::stringstream stream;
	Serialize(in, stream);
	auto out = Deserialize<decltype(in)>(stream);

	EXPECT_EQ(in, out);
}
