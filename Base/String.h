#pragma once
#include <chrono>
#include <string>

// HH:MM:SS.sss
std::string HH_MM_SS(std::chrono::duration<double>);

std::string ShortTimeString(std::chrono::duration<double> duration);

// Maps input to (..,'n', 'u', 'm', '', 'k', 'M', 'G',..)
std::string MetricPrefix(int magnitude_base_1000) noexcept(false);
