#include "pch.h"
#include "Core/Core.h"
#include <atomic>
#include <chrono>

namespace MetronomeTest
{
	using namespace std::chrono_literals;
	TEST(Metronome, Executes)
	{
		std::atomic<bool> executed = false;
		Metronome m(1ms, [&] { executed = true; });
		m.Start();
		std::this_thread::sleep_for(10ms);
		m.Stop();
		ASSERT_TRUE(executed);
	}
}