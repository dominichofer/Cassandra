#include "Core/Core.h"
#include "Search/Search.h"
#include "SearchIO/SearchIO.h"
#include "Pattern/Pattern.h"
#include "PatternIO/PatternIO.h"
#include <limits>
#include <iostream>
#include <random>
#include <vector>

void PrintHelp()
{
	std::cout
		<< "Outputs 'n' played games starting from unique positions with 'e' empty fields.\n"
		<< "   -e <int>            Number of empty fields.\n"
		<< "   -n <int>            Number of games to play.\n"
		<< "   -rnd\n"
		<< "   -self\n"
		<< "   -m1 <file>          Model 1.\n"
		<< "   -d1 <int[@float]>   Depth 1 [and confidence]\n"
		<< "   -m2 <file>          Model 2.\n"
		<< "   -d2 <int[@float]>   Depth 2 [and confidence]\n"
		<< "   -h                  Prints this help."
		<< std::endl;
}

int main(int argc, char* argv[])
{
	int empty_count = 60;
	int num = std::numeric_limits<int>::max();
	bool rnd = false;
	bool self = false;

	AAGLEM model1;
	AAGLEM model2;
	Intensity intensity1 = Intensity::Exact();
	Intensity intensity2 = Intensity::Exact();

	for (int i = 0; i < argc; i++)
	{
		if (std::string(argv[i]) == "-e") empty_count = std::stoi(argv[++i]);
		else if (std::string(argv[i]) == "-n") num = std::stoi(argv[++i]);
		else if (std::string(argv[i]) == "-rnd") rnd = true;
		else if (std::string(argv[i]) == "-self") self = true;
		else if (std::string(argv[i]) == "-m1") model1 = Deserialize<AAGLEM>(argv[++i]);
		else if (std::string(argv[i]) == "-d1") intensity1 = ParseIntensity(argv[++i]);
		else if (std::string(argv[i]) == "-m2") model2 = Deserialize<AAGLEM>(argv[++i]);
		else if (std::string(argv[i]) == "-d2") intensity2 = ParseIntensity(argv[++i]);
		else if (std::string(argv[i]) == "-h") { PrintHelp(); return 0; }
	}

	std::mt19937_64 rnd_engine(std::random_device{}());

	std::vector<Position> sample;
	std::ranges::sample(
		UniqueChildren(Position::Start(), empty_count),
		std::back_inserter(sample),
		num, rnd_engine);

	if (rnd)
	{
		for (const Game& game : RandomGamesFrom(sample))
			std::cout << to_string(game) << '\n';
	}
	else if (self)
	{
		HT tt{ 10'000'000 };
		PVS alg{ tt, model1 };
		FixedDepthPlayer player(alg, intensity1);

		for (const Game& game : SelfPlayedGamesFrom(player, sample))
			std::cout << to_string(game) << '\n';
	}
	else
	{
		HT tt1{ 10'000'000 };
		PVS alg1{ tt1, model1 };
		FixedDepthPlayer player1(alg1, intensity1);

		HT tt2{ 10'000'000 };
		PVS alg2{ tt2, model1 };
		FixedDepthPlayer player2(alg2, intensity2);

		for (const Game& game : PlayedGamesFrom(player1, player2, sample))
			std::cout << to_string(game) << '\n';
	}
	return 0;
}
