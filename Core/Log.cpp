#include "Log.h"
#include "String.h"
#include <chrono>
#include <format>
#include <iostream>

LoggingCounter::LoggingCounter(std::string name) : name(name), counter(0)
{}

LoggingCounter::~LoggingCounter()
{
	std::cout << "Counter " << name << ": " << counter << std::endl;
}

LoggingHistogram::LoggingHistogram(std::string name) : name(name)
{}

LoggingHistogram::~LoggingHistogram()
{
	std::cout << "Histogram " << name << std::endl;
	for (auto[key, value] : counter)
		std::cout << key << " " << value << std::endl;
}

static std::string HH_MM_SS(std::chrono::duration<double> duration)
{
	int hours = std::chrono::duration_cast<std::chrono::hours>(duration).count();
	int minutes = std::chrono::duration_cast<std::chrono::minutes>(duration).count() % 60;
	int seconds = std::chrono::duration_cast<std::chrono::seconds>(duration).count() % 60;
	int milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count() % 1000;

	std::string hour_str = "";
	if (hours > 0)
		hour_str = std::to_string(hours) + ":";

	std::string minute_str = "";
	if (hours > 0)
		minute_str = std::format("{:02d}:", minutes);
	else if (minutes > 0)
		minute_str = std::format("{:d}:", minutes);

	std::string second_str = "";
	if (hours > 0 || minutes > 0)
		second_str = std::format("{:02d}.", seconds);
	else
		second_str = std::format("{:d}.", seconds);

	std::string millisecond_str = std::format("{:03d}", milliseconds);

	return hour_str + minute_str + second_str + millisecond_str;
}

void Log(std::string message)
{
	using namespace std::chrono;
	auto time = floor<milliseconds>(current_zone()->to_local(system_clock::now()));
	std::cout << std::format("[{:%F %T}] {}\n", time, message);
}

void LoggingTimer::Start(std::string topic)
{
	topic_starts[topic] = std::chrono::high_resolution_clock::now();
	last_topic = topic;
}

void LoggingTimer::Stop(std::string topic)
{
	auto start = topic_starts[topic];
	auto stop = std::chrono::high_resolution_clock::now();
	Log(topic + ": " + HH_MM_SS(stop - start));
}

void LoggingTimer::Stop()
{
	Stop(last_topic);
}

Logger::Logger() : start(std::chrono::high_resolution_clock::now())
{}

void Logger::Log(std::string_view message)
{
	log(message);
}

void Logger::LogStart(std::string_view message)
{
	//log(message);
	//std::cout << '\n';
	level++;
}

void Logger::LogStop(std::string_view message)
{
	level--;
	//log(message);
	//std::cout << '\n';
}

void Logger::log(std::string_view message)
{
	//auto time = std::chrono::high_resolution_clock::now() - start;
	//std::cout << std::format("[{}] {}{}", time, std::string(level, ' '), message);
}
