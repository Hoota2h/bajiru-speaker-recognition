#pragma once

#include <gtest/gtest.h>
#include <chrono>
#include <fstream>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

/* Benchmark CSV — each run appends a row, last entry per
   test name becomes the baseline for regression detection. */
static constexpr const char* BENCH_CSV_PATH = "bench_results.csv";

/* Max allowed regression (%) before flagging.
   20% accommodates CI jitter without hiding real regressions. */
static constexpr double REGRESSION_THRESHOLD_PCT = 20.0;

/* Ignore regression % when the baseline is this small.  C++
   high_resolution_clock has nanosecond granularity in theory and
   hella jitter in practice — a 0.0001 us wobble looks like a 30%
   regression when your baseline is 0.0003 us. */
static constexpr double MIN_MEANINGFUL_BASELINE_US = 0.1;

struct BenchBaseline
{
    double meanUs = -1.0;
    bool found    = false;
};

/* Parse the CSV for previous baselines. Last entry per test wins.
   If the file doesn't exist or is garbage, you get an empty map and
   every test prints [BASELINE]. */
static std::unordered_map<std::string, BenchBaseline> loadBaselines(const std::string& csvPath)
{
    std::unordered_map<std::string, BenchBaseline> baselines;
    std::ifstream csv(csvPath);
    if (!csv.is_open())
        return baselines;

    std::string line;
    while (std::getline(csv, line))
    {
        // CSV: timestamp,test_name,mean,stddev,threshold,PASS/FAIL.
        std::istringstream ss(line);
        std::string timestamp, testName, meanStr;

        if (!std::getline(ss, timestamp, ','))
            continue;
        if (!std::getline(ss, testName, ','))
            continue;
        if (!std::getline(ss, meanStr, ','))
            continue;

        try
        {
            BenchBaseline b;
            b.meanUs            = std::stod(meanStr);
            b.found             = true;
            baselines[testName] = b; // last entry wins
        }
        catch (...)
        {
        }
    }

    return baselines;
}

// ReSharper disable once CppDFAUnreachableFunctionCall
static void appendCsv(const std::string& csvPath, const std::string& testName, const double meanUs,
                      const double stddevUs, const double thresholdUs, const bool pass)
{
    std::ofstream csv(csvPath, std::ios::app);
    if (!csv.is_open())
        return;

    const auto now = std::chrono::system_clock::now();
    const auto t   = std::chrono::system_clock::to_time_t(now);
    char timeBuf[64];
    struct tm tmBuf{};
    localtime_s(&tmBuf, &t); // thread-safe (Windows)
    std::strftime(timeBuf, sizeof(timeBuf), "%Y-%m-%dT%H:%M:%S", &tmBuf);

    csv << timeBuf << "," << testName << "," << meanUs << "," << stddevUs << "," << thresholdUs << ","
        << (pass ? "PASS" : "FAIL") << "\n";
}

static void checkBenchmark(const std::string& csvPath, const std::unordered_map<std::string, BenchBaseline>& baselines,
                           const std::string& testName, const double meanUs, const double stddevUs,
                           const double absoluteThreshold, const bool lowerIsBetter = true, const char* unit = "us")
{
    bool pass = lowerIsBetter ? (meanUs < absoluteThreshold) : (meanUs > absoluteThreshold);

    const auto it = baselines.find(testName);
    if (it != baselines.end() && it->second.found)
    {
        const double baselineMean = it->second.meanUs;
        const double pctChange    = ((meanUs - baselineMean) / baselineMean) * 100.0;

        /* Only skip % comparison for "lower is better" metrics with tiny baselines.
           "Higher is better" (throughput) baselines are large numbers — always compare. */
        const bool tooSmallToCompare = lowerIsBetter && (baselineMean < MIN_MEANINGFUL_BASELINE_US);

        if (tooSmallToCompare)
        {
            std::printf("  [OK] %s: %.4f %s (baseline: %.4f %s, too small for %% comparison)\n", testName.c_str(),
                        meanUs, unit, baselineMean, unit);
        }
        else
        {
            const bool regressed =
                lowerIsBetter ? (pctChange > REGRESSION_THRESHOLD_PCT) : (pctChange < -REGRESSION_THRESHOLD_PCT);

            if (regressed)
            {
                pass = false;
                std::printf("  [REGRESSION] %s: %.4f %s -> %.4f %s (%+.1f%%, threshold: %.0f%%)\n", testName.c_str(),
                            baselineMean, unit, meanUs, unit, pctChange, REGRESSION_THRESHOLD_PCT);
            }
            else
            {
                std::printf("  [OK] %s: %.4f %s (baseline: %.4f %s, %+.1f%%)\n", testName.c_str(), meanUs, unit,
                            baselineMean, unit, pctChange);
            }
        }
    }
    else
    {
        std::printf("  [BASELINE] %s: %.4f %s (no previous data)\n", testName.c_str(), meanUs, unit);
    }

    appendCsv(csvPath, testName, meanUs, stddevUs, absoluteThreshold, pass);

    EXPECT_TRUE(pass) << testName << " failed: mean=" << meanUs << " threshold=" << absoluteThreshold;
}

static double vecMean(const std::vector<double>& v)
{
    return std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
}

static double vecStddev(const std::vector<double>& v, const double m)
{
    double sum = 0.0;
    for (const auto& x : v)
        sum += (x - m) * (x - m);
    return std::sqrt(sum / static_cast<double>(v.size()));
}
