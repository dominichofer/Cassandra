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