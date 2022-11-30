#!/bin/sh

bin/CoreBenchmark.exe > Benchmark.log
bin/MathBenchmark.exe >> Benchmark.log
bin/PatternBenchmark.exe >> Benchmark.log
bin/SearchBenchmark.exe >> Benchmark.log