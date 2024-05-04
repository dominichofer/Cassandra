#include "Log.h"
#include "String.h"
#include <chrono>
#include <format>
#include <iostream>

void Log::Log(Severity severity, std::string_view message)
{
	switch (severity)
	{
	case Severity::Info:
		Info(message);
		break;
	case Severity::Warning:
		Warning(message);
		break;
	case Severity::Error:
		Error(message);
		break;
	}
}

void Log::Info(std::string_view message)
{
	std::cout << TimeStamp() << "[INFO] " << message << std::endl;
}

void Log::Warning(std::string_view message)
{
	std::cout << TimeStamp() << "[WARNING] " << message << std::endl;
}

void Log::Error(std::string_view message)
{
	std::cerr << TimeStamp() << "[ERROR] " << message << std::endl;
}

void Log::Log_if(bool condition, Severity severity, std::string_view message)
{
	if (condition)
		Log(severity, message);
}

void Log::Info_if(bool condition, std::string_view message)
{
	if (condition)
		Log::Info(message);
}

void Log::Warning_if(bool condition, std::string_view message)
{
	if (condition)
		Log::Warning(message);
}

void Log::Error_if(bool condition, std::string_view message)
{
	if (condition)
		Log::Error(message);
}

std::string Log::TimeStamp()
{
	using namespace std::chrono;
	auto time = floor<milliseconds>(current_zone()->to_local(system_clock::now()));
	return std::format("[{:%F %T}]", time);
}

void LoggingTimer::Start()
{
	start = std::chrono::high_resolution_clock::now();
}

void LoggingTimer::Stop(std::string_view message)
{
	auto stop = std::chrono::high_resolution_clock::now();
	Log::Log(severity, std::format("{} ({})", message, HH_MM_SS(stop - start)));
}
