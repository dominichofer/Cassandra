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
		std::size_t result = HashTableMap::perft(d, RAM);
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