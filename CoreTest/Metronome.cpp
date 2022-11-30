#include "pch.h"
#include <atomic>
#include <chrono>
#include <thread>

namespace MetronomeTest
{
	using namespace std::chrono_literals;
	TEST(Metronome, Executes)
	{
		std::atomic<bool> executed = false;
		Metronome m(1ms, [&] { executed.store(true, std::memory_order_release); });
		m.Start();
		std::this_thread::sleep_for(100ms);
		m.Stop();
		ASSERT_TRUE(executed.load(std::memory_order_acquire));
	}
}