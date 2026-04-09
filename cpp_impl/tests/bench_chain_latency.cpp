#include <gtest/gtest.h>
#include "Constants.h"
#include "SharedRingBuffer.h"
#include "RmsAnalyzer.h"
#include "bench_csv.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <string>
#include <thread>
#include <vector>

// Production values — keep benchmarks in sync with real config.
static constexpr int CAPACITY = linkjiru::ringBufferCapacity;
static constexpr int WINDOW   = linkjiru::defaultAnalysisWindow;

/* Absolute latency ceilings (us).  Benchmark fails if mean exceeds these.
   - Input:   write + readLastN round-trip.
   - Compute: single isSpeechActive() call.
   - Output:  state-machine update + atomic store.
   - Full:    all three combined. */
static constexpr double MAX_INPUT_LATENCY_US   = 20'000.0;
static constexpr double MAX_COMPUTE_LATENCY_US = 100.0;
static constexpr double MAX_OUTPUT_LATENCY_US  = 10.0;
static constexpr double MAX_FULL_CHAIN_US      = 25'000.0;

static constexpr int BENCH_ITERATIONS = 10000;

/* Mirrors AnalysisThread::Config defaults for the state-machine
   simulation in OutputLatency / FullChainLatency. */
static constexpr int SUSTAIN_MS       = 100; // speech-sustain hold
static constexpr int POLL_INTERVAL_MS = 16;  // ~60 fps fake clock tick

// Typical DAW block size for write-path benchmarks.
static constexpr int WRITE_BLOCK_SIZE = 512;

using Clock = std::chrono::high_resolution_clock;

static std::vector<float> generateSilence(const int count)
{
    return std::vector<float>(count, 0.001f);
}

static std::vector<float> generateSpeech(const int count)
{
    std::vector<float> samples(count);
    for (int i = 0; i < count; ++i)
        samples[i] = 0.5f * std::sin(static_cast<float>(i) * 0.3f);
    return samples;
}

class ChainLatencyBench : public ::testing::Test
{
protected:
    SharedRingBuffer<CAPACITY> buffer;
    std::unordered_map<std::string, BenchBaseline> baselines;

    void SetUp() override { baselines = loadBaselines(BENCH_CSV_PATH); }
};

TEST_F(ChainLatencyBench, ComputeLatency)
{
    RmsAnalyzer::Config cfg;
    cfg.calibrationSamples = WINDOW;
    RmsAnalyzer analyzer(cfg);

    const auto silence = generateSilence(WINDOW);
    analyzer.calibrate(silence.data(), WINDOW);
    ASSERT_TRUE(analyzer.isCalibrated());

    const auto speech = generateSpeech(WINDOW);
    std::vector<double> timings;

    for (int iter = 0; iter < BENCH_ITERATIONS; ++iter)
    {
        auto start           = Clock::now();
        volatile bool result = analyzer.isSpeechActive(speech.data(), WINDOW);
        auto end             = Clock::now();
        (void)result;

        timings.push_back(std::chrono::duration<double, std::micro>(end - start).count());
    }

    const double m = vecMean(timings);
    const double s = vecStddev(timings, m);
    checkBenchmark(BENCH_CSV_PATH, baselines, "ComputeLatency_RMS_2048", m, s, MAX_COMPUTE_LATENCY_US);
}

TEST_F(ChainLatencyBench, OutputLatency)
{
    std::atomic<float> detectValue{0.0f};
    bool currentlySpeaking = false;
    int64_t lastSpeechTime = 0;
    std::vector<double> timings;

    for (int iter = 0; iter < BENCH_ITERATIONS; ++iter)
    {
        const bool speechDetected = (iter % 2 == 0);
        const auto now            = static_cast<int64_t>(iter * POLL_INTERVAL_MS);

        auto start = Clock::now();

        // Mirror production (AnalysisThread::run) — no guard on the assignment.
        if (speechDetected)
        {
            lastSpeechTime    = now;
            currentlySpeaking = true;
        }
        else if (currentlySpeaking && (now - lastSpeechTime > SUSTAIN_MS))
        {
            currentlySpeaking = false;
        }

        detectValue.store(currentlySpeaking ? 1.0f : 0.0f);

        auto end = Clock::now();

        timings.push_back(std::chrono::duration<double, std::micro>(end - start).count());
    }

    const double m = vecMean(timings);
    const double s = vecStddev(timings, m);
    checkBenchmark(BENCH_CSV_PATH, baselines, "OutputLatency_StateMachine", m, s, MAX_OUTPUT_LATENCY_US);
}

TEST_F(ChainLatencyBench, InputLatency)
{
    /* Single-threaded write → readLastN round-trip.
       Just the raw transport cost. Real-world
       adds one poll interval (~16ms) but that's a known constant, unlike everything
       else in concurrent C++. */
    const auto silence = generateSilence(CAPACITY / 2);
    buffer.write(silence.data(), CAPACITY / 2);

    const auto speech = generateSpeech(WRITE_BLOCK_SIZE);
    std::vector<float> dest(WINDOW);
    std::vector<double> timings;

    for (int iter = 0; iter < 1000; ++iter)
    {
        auto start = Clock::now();
        buffer.write(speech.data(), WRITE_BLOCK_SIZE);
        buffer.readLastN(dest.data(), WINDOW);
        auto end = Clock::now();

        timings.push_back(std::chrono::duration<double, std::micro>(end - start).count());
    }

    const double m = vecMean(timings);
    const double s = vecStddev(timings, m);
    checkBenchmark(BENCH_CSV_PATH, baselines, "InputLatency_Write_to_Read", m, s, MAX_INPUT_LATENCY_US);
}

TEST_F(ChainLatencyBench, FullChainLatency)
{
    RmsAnalyzer::Config cfg;
    cfg.calibrationSamples = WINDOW;
    RmsAnalyzer analyzer(cfg);

    const auto silence = generateSilence(WINDOW);
    buffer.write(silence.data(), WINDOW);

    std::vector<float> windowBuf(WINDOW);
    buffer.readLastN(windowBuf.data(), WINDOW);
    analyzer.calibrate(windowBuf.data(), WINDOW);
    ASSERT_TRUE(analyzer.isCalibrated());

    const auto speech = generateSpeech(WRITE_BLOCK_SIZE);
    std::atomic<float> detectValue{0.0f};
    bool currentlySpeaking = false;
    int64_t lastSpeechTime = 0;

    std::vector<double> timings;

    for (int iter = 0; iter < 100; ++iter)
    {
        auto start = Clock::now();

        buffer.write(speech.data(), WRITE_BLOCK_SIZE);
        buffer.readLastN(windowBuf.data(), WINDOW);
        const bool speechDetected = analyzer.isSpeechActive(windowBuf.data(), WINDOW);

        const auto now = static_cast<int64_t>(iter * POLL_INTERVAL_MS);
        if (speechDetected)
        {
            lastSpeechTime    = now;
            currentlySpeaking = true;
        }
        else if (currentlySpeaking && (now - lastSpeechTime > SUSTAIN_MS))
        {
            currentlySpeaking = false;
        }

        detectValue.store(currentlySpeaking ? 1.0f : 0.0f);

        auto end = Clock::now();

        timings.push_back(std::chrono::duration<double, std::micro>(end - start).count());
    }

    const double m = vecMean(timings);
    const double s = vecStddev(timings, m);
    checkBenchmark(BENCH_CSV_PATH, baselines, "FullChainLatency", m, s, MAX_FULL_CHAIN_US);
}
