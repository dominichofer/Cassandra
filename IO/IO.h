#pragma once
#include "Binary.h"
#include "File.h"
#include "Integers.h"
#include "ResultTable.h"
#include "String.h"
#include "Table.h"
#include <string>
#include <chrono>

std::string short_time_format(std::chrono::duration<double> duration);
