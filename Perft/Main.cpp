#include "Hashtable.h"
#include "Perft.h"
#include <chrono>
#include <iostream>
#include <string>
#include <sstream>
#include <iomanip>

std::size_t ParseBytes(const std::string& bytes)
{
	if (bytes.find("EB") != std::string::npos) return std::stoll(bytes) * 1024 * 1024 * 1024 * 1024 * 1024 * 1024;
	if (bytes.find("PB") != std::string::npos) return std::stoll(bytes) * 1024 * 1024 * 1024 * 1024 * 1024;
	if (bytes.find("TB") != std::string::npos) return std::stoll(bytes) * 1024 * 1024 * 1024 * 1024;
	if (bytes.find("GB") != std::string::npos) return std::stoll(bytes) * 1024 * 1024 * 1024;
	if (bytes.find("MB") != std::string::npos) return std::stoll(bytes) * 1024 * 1024;
	if (bytes.find("kB") != std::string::npos) return std::stoll(bytes) * 1024;
	if (bytes.find( 'B') != std::string::npos) return std::stoll(bytes);
	return 0;
}

void PrintHelp()
{
	std::cout
		<< "   -d    Depth of perft.\n"
		<< "   -RAM  Number of hash table bytes.\n"
		<< "   -h    Prints this help."
		<< std::endl;
}

std::string ThousandsSeparator(uint64_t n)
{
	std::ostringstream oss;
	std::locale locale("");
	oss.imbue(locale);
	oss << n;
	return oss.str();
}

//ddd:hh:mm:ss.ccc
std::string time_format(const std::chrono::milliseconds duration)
{
	using days_t = std::chrono::duration<int, std::ratio<24 * 3600> >;
	const auto millis = duration.count() % 1000;
	const auto seconds = std::chrono::duration_cast<std::chrono::seconds>(duration).count() % 60;
	const auto minutes = std::chrono::duration_cast<std::chrono::minutes>(duration).count() % 60;
	const auto hours = std::chrono::duration_cast<std::chrono::hours>  (duration).count() % 24;
	const auto days = std::chrono::duration_cast<days_t>              (duration).count();

	std::ostringstream oss;
	oss << std::setfill(' ');

	if (days != 0)
		oss << std::setw(3) << days << ":" << std::setfill('0');
	else
		oss << "    ";

	if ((days != 0) || (hours != 0))
		oss << std::setw(2) << hours << ":" << std::setfill('0');
	else
		oss << "   ";

	if ((days != 0) || (hours != 0) || (minutes != 0))
		oss << std::setw(2) << minutes << ":" << std::setfill('0');
	else
		oss << "   ";

	oss << std::setw(2) << seconds << "." << std::setfill('0') << std::setw(3) << millis;

	return oss.str();
}

#include <numeric>
#include <functional>
#include <execution>
#include "Core/PositionGenerator.h"

std::size_t Number_of_different_positions(const std::vector<Position>& all)
{
	return std::inner_product(all.begin() + 1, all.end(), all.begin(), 1, std::plus(), std::not_equal_to());
}

// Counts Othello positions that occure once and only once in the list.
std::size_t Number_of_unique_realization(const std::vector<Position>& all)
{
	const std::size_t size = all.size();
	if (size < 2)
		return size;

	int64_t sum = (all[0] != all[1]) ? 1 : 0;
	#pragma omp parallel for reduction(+:sum)
	for (int64_t i = 1; i < size - 1; i++)
		if ((all[i-1] != all[i]) && (all[i] != all[i+1]))
			sum++;
	if (all[size - 2] != all[size - 1])
		sum++;
	return sum;
}

//int main()
//{
//	printf("plies|     A124005 |     A124006 |     A125528 |     A125529 |      time \n");
//	printf("-----+-------------+-------------+-------------+-------------+-----------\n");
//
//	for (int plies = 0; plies < 20; plies++)
//	{
//		const auto start = std::chrono::high_resolution_clock::now();
//
//		std::vector<Position> all;
//		generate_all(std::back_inserter(all), PosGen::All_after_nth_ply(plies, 1));
//		std::sort(std::execution::par, all.begin(), all.end());
//
//		const auto end = std::chrono::high_resolution_clock::now();
//		const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
//		const auto milliseconds = duration.count();
//
//		printf(" %3u", plies);
//		printf(" | %11s", ThousandsSeparator(Number_of_different_positions(all)).c_str());
//		printf(" | %11s", ThousandsSeparator(Number_of_unique_realization(all)).c_str());
//		std::transform(std::execution::par, all.begin(), all.end(), all.begin(), 
//					   [](const Position& pos) { return Position(pos.P | pos.O, 0); });
//		std::sort(std::execution::par, all.begin(), all.end());
//		printf(" | %11s", ThousandsSeparator(Number_of_different_positions(all)).c_str());
//		printf(" | %11s", ThousandsSeparator(Number_of_unique_realization(all)).c_str());
//		printf(" | % 9s\n", time_format(duration).c_str());
//	}
//	return 0;
//}

//plies|        A124005 |        A124006 |             time
//-----+----------------+----------------+------------------
//   0 |              1 |              1 |            0.000
//   1 |              4 |              4 |            0.000
//   2 |             12 |             12 |            0.000
//   3 |             54 |             52 |            0.000
//   4 |            236 |            228 |            0.000
//   5 |          1'288 |          1'192 |            0.000
//   6 |          7'092 |          6'160 |            0.000
//   7 |         42'614 |         33'344 |            0.001
//   8 |        269'352 |        191'380 |            0.015
//   9 |      1'743'560 |      1'072'232 |            0.115
//  10 |     11'922'442 |      6'416'600 |            0.967
//  11 |     80'209'268 |     35'990'544 |            9.097
//  12 |    562'280'115 |    212'278'256 |         1:20.681

int main(int argc, char* argv[])
{
	int depth = 20;
	std::size_t RAM = 1024 * 1024 * 1024;

	for (int i = 0; i < argc; i++)
	{
		if (std::string(argv[i]) == "-d") depth = atoi(argv[++i]);
		else if (std::string(argv[i]) == "-RAM") RAM = ParseBytes(argv[++i]);
		else if (std::string(argv[i]) == "-h") { PrintHelp(); return 0; }
	}

	std::cout << "depth|       Positions        |correct|       Time       |       N/s       " << std::endl;
	std::cout << "-----+------------------------+-------+------------------+-----------------" << std::endl;

	std::chrono::high_resolution_clock::time_point startTime, endTime;
	for (uint8_t d = 1; d <= depth; d++)
	{
		const auto start = std::chrono::high_resolution_clock::now();
		std::size_t result = HashTableMap::perft(d, 100'000'000);
		const auto end = std::chrono::high_resolution_clock::now();
		const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
		const auto milliseconds = duration.count();

		printf(" %3u | %22s |%7s| %14s | %15s\n",
			d,
			ThousandsSeparator(result).c_str(), (Correct(d) == result ? "  true " : " false "),
			time_format(duration).c_str(),
			milliseconds > 0 ? ThousandsSeparator(result / milliseconds * 1000).c_str() : ""
		);
	}

	return 0;
}