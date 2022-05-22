#include "Metronome.h"

void Metronome::CallbackLoop(std::stop_token token)
{
	std::unique_lock lock(mutex);
	while (!token.stop_requested())
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

void Metronome::Force()
{
	std::unique_lock lock(mutex);
	callback();
}
