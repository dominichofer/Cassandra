#pragma once
#include <chrono>
#include <string>
#include <string_view>

namespace Log
{
	enum class Severity
	{
		Info, Warning, Error
	};

	void Log(Severity, std::string_view message);
	void Info(std::string_view message);
	void Warning(std::string_view message);
	void Error(std::string_view message);

	void Log_if(bool condition, Severity, std::string_view message);
	void Info_if(bool condition, std::string_view message);
	void Warning_if(bool condition, std::string_view message);
	void Error_if(bool condition, std::string_view message);

	std::string TimeStamp();
}

class LoggingTimer
{
	std::chrono::high_resolution_clock::time_point start;
	Log::Severity severity;
public:
	LoggingTimer(Log::Severity severity = Log::Severity::Info)
		: start(std::chrono::high_resolution_clock::now())
		, severity(severity)
	{}

	void Start();
	void Stop(std::string_view message);
};
