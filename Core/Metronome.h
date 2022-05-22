#pragma once
#include <chrono>
#include <condition_variable>
#include <functional>
#include <thread>

class Metronome
{
	std::mutex mutex;
	std::condition_variable cv;
	std::chrono::duration<double> period;
	std::function<void()> callback;
	std::jthread thread;

	void CallbackLoop(std::stop_token);
public:
	Metronome(std::chrono::duration<double> period, std::function<void()> callback) noexcept
		: period(period), callback(std::move(callback))
	{}
	~Metronome() { Stop(); }

	void Start();
	void Stop();
	void Force();
	bool Runs() const { return thread.joinable(); }
};


//class StopWatch
//{
//	std::chrono::high_resolution_clock::time_point start;
//	std::vector<std::chrono::high_resolution_clock::duration> laps;
//public:
//	StopWatch() = default;
//	static StopWatch Started() { StopWatch w; w.Start(); return w; }
//	void Start() { start = std::chrono::high_resolution_clock::now(); }
//	void Lap() { laps.push_back(std::chrono::high_resolution_clock::now() - start); }
//	const auto& Laps() const { return laps; }
//};