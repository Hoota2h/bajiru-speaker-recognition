#include <gtest/gtest.h>
#include "Constants.h"
#include "SharedRingBuffer.h"
#include "bench_csv.h"

#include <chrono>
#include <cmath>
#include <string>
#include <vector>

// Production values.
static constexpr int CAPACITY = linkjiru::ringBufferCapacity;
static constexpr int WINDOW   = linkjiru::defaultAnalysisWindow;

static constexpr int ITERATIONS = 1000;

/* Performance ceilings — benchmark fails if breached.
   - Write:   max us per sample.
   - Read:    max us for one readLastN snapshot.
   - SeqRead: min samples/sec for sequential cursor reads. */
static constexpr double MAX_WRITE_US_PER_SAMPLE  = 1.0;
static constexpr double MAX_READ_LAST_N_US       = 50.0;
static constexpr double MIN_SEQ_READ_SAMPLES_SEC = 100'000'000.0;

using Clock = std::chrono::high_resolution_clock;

class RingBufferBench : public ::testing::Test
{
protected:
    SharedRingBuffer<CAPACITY> buffer;
    std::unordered_map<std::string, BenchBaseline> baselines;

    void SetUp() override { baselines = loadBaselines(BENCH_CSV_PATH); }

    static std::vector<float> generateSamples(const int count)
    {
        std::vector<float> samples(count);
        for (int i = 0; i < count; ++i)
            samples[i] = std::sin(static_cast<float>(i) * 0.1f);
        return samples;
    }

    void benchWrite(const int blockSize)
    {
        const auto samples = generateSamples(blockSize);
        std::vector<double> timings;

        for (int iter = 0; iter < ITERATIONS; ++iter)
        {
            auto start = Clock::now();
            buffer.write(samples.data(), blockSize);
            auto end = Clock::now();

            timings.push_back(std::chrono::duration<double, std::micro>(end - start).count() /
                              static_cast<double>(blockSize));
        }

        const double m         = vecMean(timings);
        const double s         = vecStddev(timings, m);
        const std::string name = "WriteThroughput_" + std::to_string(blockSize);
        checkBenchmark(BENCH_CSV_PATH, baselines, name, m, s, MAX_WRITE_US_PER_SAMPLE, true, "us/sample");
    }
};

TEST_F(RingBufferBench, WriteThroughput_64)
{
    benchWrite(64);
}
TEST_F(RingBufferBench, WriteThroughput_512)
{
    benchWrite(512);
}
TEST_F(RingBufferBench, WriteThroughput_2048)
{
    benchWrite(2048);
}

TEST_F(RingBufferBench, ReadLastN_Latency)
{
    const auto samples = generateSamples(WINDOW);
    for (int i = 0; i < 100; ++i)
        buffer.write(samples.data(), WINDOW);

    std::vector<float> dest(WINDOW);
    std::vector<double> timings;

    for (int iter = 0; iter < ITERATIONS; ++iter)
    {
        auto start = Clock::now();
        buffer.readLastN(dest.data(), WINDOW);
        auto end = Clock::now();

        timings.push_back(std::chrono::duration<double, std::micro>(end - start).count());
    }

    const double m = vecMean(timings);
    const double s = vecStddev(timings, m);
    checkBenchmark(BENCH_CSV_PATH, baselines, "ReadLastN_Latency_2048", m, s, MAX_READ_LAST_N_US);
}

TEST_F(RingBufferBench, SequentialRead_Throughput)
{
    const auto samples = generateSamples(4096);
    for (int i = 0; i < 64; ++i)
        buffer.write(samples.data(), 4096);

    uint64_t cursor = 0;
    float dest[4096];
    int64_t totalSamples = 0;

    const auto start = Clock::now();
    for (int iter = 0; iter < ITERATIONS; ++iter)
    {
        const auto result = buffer.read(dest, 4096, cursor);
        totalSamples += result.samplesRead;

        if (result.samplesRead == 0)
            buffer.write(samples.data(), 4096);
    }
    const auto end = Clock::now();

    const double seconds       = std::chrono::duration<double>(end - start).count();
    const double samplesPerSec = static_cast<double>(totalSamples) / seconds;

    checkBenchmark(BENCH_CSV_PATH, baselines, "SequentialRead_Throughput", samplesPerSec / 1e6, 0.0,
                   MIN_SEQ_READ_SAMPLES_SEC / 1e6, false, "M samples/sec");
}

TEST_F(RingBufferBench, OverrunRecovery)
{
    uint64_t cursor = 0;
    float dest[256];

    const auto samples = generateSamples(256);
    buffer.write(samples.data(), 256);
    buffer.read(dest, 256, cursor);
    EXPECT_EQ(cursor, 256u);

    const auto bigBlock = generateSamples(4096);
    for (int i = 0; i < 40; ++i)
        buffer.write(bigBlock.data(), 4096);

    const auto result = buffer.read(dest, 256, cursor);
    EXPECT_TRUE(result.overrun);
    EXPECT_GT(cursor, 256u);
    EXPECT_GT(result.samplesRead, 0);
}

TEST_F(RingBufferBench, MultiReaderIndependence)
{
    uint64_t cursorA = 0;
    uint64_t cursorB = 0;
    float destA[512];
    float destB[512];

    const auto samples = generateSamples(1024);
    buffer.write(samples.data(), 1024);

    const auto resultA = buffer.read(destA, 512, cursorA);
    EXPECT_EQ(resultA.samplesRead, 512);
    EXPECT_EQ(cursorA, 512u);

    const auto resultB = buffer.read(destB, 512, cursorB);
    EXPECT_EQ(resultB.samplesRead, 512);
    EXPECT_EQ(cursorB, 512u);

    for (int i = 0; i < 512; ++i)
        EXPECT_FLOAT_EQ(destA[i], destB[i]);

    [[maybe_unused]] auto resultA2 = buffer.read(destA, 512, cursorA);
    EXPECT_EQ(cursorA, 1024u);
    EXPECT_EQ(cursorB, 512u);
}
