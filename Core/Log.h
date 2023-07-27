#pragma once
#include <chrono>
#include <map>
#include <string>
#include <string_view>

class LoggingCounter
{
	std::string name;
	uint64_t counter;
public:
	LoggingCounter(std::string name);
	~LoggingCounter();

	void operator++() { ++counter; }
	void operator++(int) { counter++; }
	void Count(uint64_t num = 1) { counter += num; }
};

class LoggingHistogram
{
	std::chrono::high_resolution_clock::time_point start;
	std::string name;
	std::map<uint64_t, uint64_t> counter;
public:
	LoggingHistogram(std::string name);
	~LoggingHistogram();

	auto& operator[](int index) { return counter[index]; }
};

void Log(std::string message);

class LoggingTimer
{
	std::map<std::string, std::chrono::steady_clock::time_point> topic_starts;
	std::string last_topic;
public:
	void Start(std::string topic);
	void Stop(std::string topic);
	void Stop();
};

class Logger
{
	std::chrono::high_resolution_clock::time_point start;
	int level = 0;
public:
	Logger();

	void Log(std::string_view);
	void LogStart(std::string_view);
	void LogStop(std::string_view);
private:
	void log(std::string_view);
};

extern inline Logger logging{};