#include "IO/IO.h"
#include "Math/Statistics.h"
#include "Core/Core.h"
#include <iostream>
#include <fstream>
#include <execution>
#include <algorithm>


#pragma pack(1)
struct PosResDur
{
	Position pos;
	uint64 value;
	double duration;

	bool operator==(const PosResDur& o) const { return pos == o.pos; }
	bool operator<(const PosResDur& o) const { return pos < o.pos; }
};
#pragma pack()

std::vector<PosResDur> ReadFile(const std::filesystem::path& path)
{
	std::vector<PosResDur> content;
	std::fstream stream(path, std::ios::in | std::ios::binary);
	auto eof = [&](){ stream.peek(); return stream.eof(); };
	while (stream && not eof())
	{
		PosResDur buffer;
		stream.read(reinterpret_cast<char*>(&buffer), sizeof(PosResDur));
		content.push_back(buffer);
	}
	stream.close();
	return content;
}

int main()
{
	//for (int i = 6; i < 7; i++)
	//{
	//	std::vector<Position> v;
	//	for (auto pos : Children(Position::Start(), i, true))
	//		v.push_back(FlipToUnique(pos));
	//	std::sort(v.begin(), v.end());
	//	auto it = std::unique(v.begin(), v.end());
	//	for (auto pos : v)
	//		std::cout << "![Image](https://raw.githubusercontent.com/PanicSheep/ReversiPerftCUDA/master/docs/"
	//			<< SingleLine(pos) << ",png) " << 
	//	std::cout << i << " " << std::distance(v.begin(), it) << std::endl;
	//}
	//for (int i = 6; i < 7; i++)
	//{
		//std::vector<PosRes> pos;
		//for (const auto& p : Children(Position::Start(), i, true))
		//	pos.push_back({FlipToUnique(p), 0});
		//std::sort(std::execution::par_unseq, pos.begin(), pos.end());
		//auto last = std::unique(std::execution::par_unseq, pos.begin(), pos.end());
		//pos.erase(last, pos.end()); 
		//std::cout << i << ": " << pos.size() << std::endl;
		//WriteToFile(R"(G:\Reversi\perft\ply)" + std::to_string(i) + ".pos", pos);

		std::locale locale("");
		std::cout.imbue(locale);

		std::vector<PosResDur> results = ReadFile(R"(G:\Reversi\perft\perft20_ply6.pos)");
		//uint64 sum = 0;
		int counter = 0;
		for (const auto& p : results/*Children(Position::Start(), 6, true)*/)
		{
			//uint64 value = std::lower_bound(results.begin(), results.end(), PosRes{FlipToUnique(p), 0})->value;
			std::cout << "|" << counter++ << "|![Image](https://raw.githubusercontent.com/PanicSheep/ReversiPerftCUDA/master/docs/ply6/"
				<< SingleLine(p.pos).substr(0, 64) << ".png)|" << p.value << "|" << std::endl;
			//sum += value;
		}
		//std::cout << sum;
		//std::cout << i << ": " << pos.size() << std::endl;
		//WriteToFile(R"(G:\Reversi\perft\ply)" + std::to_string(i) + ".pos", pos);
		

		//std::vector<PosScore> data;
		//ReadPosScoreFile(std::back_inserter(data), input);
		//std::cout << i << ": " << SampleStandardDeviation(data, [](const auto& x) { return x.score; }) << "\n";
	//}
	return 0;
}