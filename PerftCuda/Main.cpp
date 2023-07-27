#include "kernel.cuh"
#include "Perft/Perft.h"
#include "PerftCuda.h"
#include <chrono>
#include <string>
#include <iostream>
#include <iomanip>
#include <omp.h>

void PrintHelp()
{
	std::cout
		<< "   -d     Depth of perft.\n"
		<< "   -RAM   Number of hash table bytes.\n"
		<< "   -cuda  To run on cuda capable GPUs.\n"
		<< "   -h     Prints this help."
		<< std::endl;
}

void ResetCudaDevices()
{
	int n;
	cudaGetDeviceCount(&n);
	for (int i = 0; i < n; i++)
	{
		cudaSetDevice(i);
		cudaError_t cudaStatus = cudaDeviceReset();
		if (cudaStatus != cudaSuccess)
			fprintf(stderr, "cudaDeviceReset failed!");
	}
}

int main(int argc, char* argv[])
{
	ResetCudaDevices();

	int depth = 17;
	std::size_t RAM = 48'000'000'000;
	bool cuda = true;

	for (int i = 0; i < argc; i++)
	{
		if (std::string(argv[i]) == "-d") depth = std::stoi(argv[++i]);
		//else if (std::string(argv[i]) == "-RAM") RAM = ParseBytes(argv[++i]);
		else if (std::string(argv[i]) == "-cuda") cuda = true;
		else if (std::string(argv[i]) == "-h") { PrintHelp(); return 0; }
	}

	std::unique_ptr<BasicPerft> engine;
	if (cuda && RAM)
		engine = std::make_unique<CudaHashTablePerft>(RAM, 5, 5, 3);
	else if (RAM)
		engine = std::make_unique<HashTablePerft>(RAM, 6);
	else
		engine = std::make_unique<UnrolledPerft>(6);

	std::cout << "depth|       Positions       |correct|      Time[s]     |       Pos/s      \n";
	std::cout << "-----+-----------------------+-------+------------------+------------------\n";

	std::cout.imbue(std::locale(""));
	std::cout << std::setfill(' ') << std::boolalpha;

	for (int d = 15; d <= depth; d++)
	{
		engine->clear();
		const auto start = std::chrono::high_resolution_clock::now();
		const auto result = engine->calculate(d);
		const auto end = std::chrono::high_resolution_clock::now();
		const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1'000.0;

		std::cout << std::setw(4) << d << " |";
		std::cout << std::setw(22) << result << " |";
		std::cout << std::setw(6) << (Correct(d) == result) << " |";
		std::cout << std::setw(17) << duration << " |";
		if (duration)
			std::cout << std::setw(17) << static_cast<int64_t>(result / duration);
		std::cout << '\n';
	}

	ResetCudaDevices();
    return 0;
}
