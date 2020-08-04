#include <iostream>
#include <fstream>
#include "IO/IO.h"

#pragma pack(1)
struct DensePositionScore
{
	Position pos;
	int8_t score;
};
#pragma pack(pop)

template <typename Vector>
double SampleStandardDeviation(const Vector& vec)
{
	double E_of_X = 0;
	double E_of_X_sq = 0;
	for (std::size_t i = 0; i < vec.size(); i++)
	{
		const double x = vec[i].score;
		const double N = static_cast<double>(i + 1);
		E_of_X += (x - E_of_X) / N;
		E_of_X_sq += (x * x - E_of_X_sq) / N;
	}
	return std::sqrt(E_of_X_sq - E_of_X * E_of_X);
}

int main()
{
	for (int i = 0; i < 61; i++)
	{
		const auto input = R"(G:\Reversi\rnd\e)" + std::to_string(i) + ".psc";

		std::vector<DensePositionScore> data;
		ReadFile<DensePositionScore>(std::back_inserter(data), input);
		std::cout << i << ": " << SampleStandardDeviation(data) << "\n";
	}
	return 0;
}