#include "Metronome.h"

void Metronome::CallbackLoop(std::stop_token token)
{
	std::mutex mutex;
	std::unique_lock<std::mutex> lock(mutex);
	while (true)
	{
		cv.wait_for(lock, period);
		if (token.stop_requested())
			break;
		callback();
	}
}

void Metronome::Start()
{
	thread = std::jthread([this](std::stop_token token) { CallbackLoop(token); });
}

void Metronome::Stop()
{
	if (thread.joinable())
	{
		thread.request_stop();
		cv.notify_all();
		thread.join();
	}
}
