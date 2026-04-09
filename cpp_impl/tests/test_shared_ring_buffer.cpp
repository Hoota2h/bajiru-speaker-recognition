#include <gtest/gtest.h>
#include "Constants.h"
#include "SharedRingBuffer.h"
#include "PluginProcessor.h"
#include "mock_vts_server.h"

#include <atomic>
#include <cmath>
#include <thread>
#include <vector>

#define TAG (::testing::Test::HasFailure() ? "FAIL" : "PASS")

TEST(RingBuffer_WrapAround, Capacity4)
{
    SharedRingBuffer<4> buf;

    float src[10];
    for (int i = 0; i < 10; ++i)
        src[i] = static_cast<float>(i + 1);

    buf.write(src, 10);

    float dest[4] = {};
    buf.readLastN(dest, 4);

    EXPECT_FLOAT_EQ(dest[0], 7.0f);
    EXPECT_FLOAT_EQ(dest[1], 8.0f);
    EXPECT_FLOAT_EQ(dest[2], 9.0f);
    EXPECT_FLOAT_EQ(dest[3], 10.0f);

    std::printf("  [%s] Capacity4: wrote 10, last 4 = {%.0f,%.0f,%.0f,%.0f}\n", TAG, dest[0], dest[1], dest[2],
                dest[3]);
}

TEST(RingBuffer_WrapAround, Capacity16)
{
    SharedRingBuffer<16> buf;

    float src[50];
    for (int i = 0; i < 50; ++i)
        src[i] = static_cast<float>(i);

    buf.write(src, 50);

    float dest[16] = {};
    buf.readLastN(dest, 16);

    for (int i = 0; i < 16; ++i)
        EXPECT_FLOAT_EQ(dest[i], static_cast<float>(34 + i));

    std::printf("  [%s] Capacity16: wrote 50, last 16 = [%.0f..%.0f] (3.1x wrap)\n", TAG, dest[0], dest[15]);
}

TEST(RingBuffer_WrapAround, Capacity16_MultipleWrites)
{
    SharedRingBuffer<16> buf;

    for (int batch = 0; batch < 20; ++batch)
    {
        float chunk[3];
        for (int i = 0; i < 3; ++i)
            chunk[i] = static_cast<float>(batch * 3 + i);
        buf.write(chunk, 3);
    }

    float dest[16] = {};
    buf.readLastN(dest, 16);

    for (int i = 0; i < 16; ++i)
        EXPECT_FLOAT_EQ(dest[i], static_cast<float>(44 + i));

    std::printf("  [%s] Capacity16_MultipleWrites: 20x3 batches, last 16 = [%.0f..%.0f]\n", TAG, dest[0], dest[15]);
}

// Overrun: the reader fell asleep and the writer doesn't care

TEST(RingBuffer_Overrun, DetectsOverrunAndRecovers)
{
    SharedRingBuffer<16> buf;

    uint64_t cursor = 0;
    float dest[4];

    float initial[4] = {1, 2, 3, 4};
    buf.write(initial, 4);
    auto r = buf.read(dest, 4, cursor);
    EXPECT_EQ(r.samplesRead, 4);
    EXPECT_FALSE(r.overrun);
    EXPECT_EQ(cursor, 4u);

    for (int i = 0; i < 5; ++i)
    {
        float block[4];
        for (int j = 0; j < 4; ++j)
            block[j] = static_cast<float>(100 + i * 4 + j);
        buf.write(block, 4);
    }

    const auto cursorBefore = cursor;
    r                       = buf.read(dest, 4, cursor);
    EXPECT_TRUE(r.overrun);
    EXPECT_GT(cursor, 4u);
    EXPECT_GT(r.samplesRead, 0);

    r = buf.read(dest, 4, cursor);
    EXPECT_FALSE(r.overrun);

    std::printf("  [%s] DetectsOverrunAndRecovers: cursor %llu -> %llu, "
                "post-recovery read clean\n",
                TAG, static_cast<unsigned long long>(cursorBefore), static_cast<unsigned long long>(cursor));
}

TEST(RingBuffer_Overrun, PostRecoveryDataIsValid)
{
    SharedRingBuffer<16> buf;

    float seq[32];
    for (int i = 0; i < 32; ++i)
        seq[i] = static_cast<float>(i);
    buf.write(seq, 32);

    uint64_t cursor = 0;
    float dest[8];
    auto r = buf.read(dest, 8, cursor);
    EXPECT_TRUE(r.overrun);

    float minVal = dest[0], maxVal = dest[0];
    for (int i = 0; i < r.samplesRead; ++i)
    {
        EXPECT_GE(dest[i], 16.0f);
        EXPECT_LE(dest[i], 31.0f);
        minVal = std::min(minVal, dest[i]);
        maxVal = std::max(maxVal, dest[i]);
    }

    std::printf("  [%s] PostRecoveryDataIsValid: %d samples in [%.0f..%.0f] "
                "(valid: 16-31)\n",
                TAG, r.samplesRead, minVal, maxVal);
}

// "thread-safe"?

TEST(RingBuffer_Concurrent, RealPluginProducerConsumer)
{
    MockVtsServer mockVts;
    mockVts.start();
    const auto port = std::to_string(mockVts.getPort());

    LinkjiruProcessor processor;

    /* 192 kHz / 32-sample blocks — way above production rates
       to stress the ring buffer under high write pressure.
       Good stress-test values */
    static constexpr double SAMPLE_RATE        = 192000.0;
    static constexpr int BLOCK_SIZE            = 32;
    static constexpr int TEST_DURATION_SECONDS = 4;
    static constexpr int TOTAL_BLOCKS          = static_cast<int>((SAMPLE_RATE * TEST_DURATION_SECONDS) / BLOCK_SIZE);

    processor.prepareToPlay(SAMPLE_RATE, BLOCK_SIZE);
    processor.startAnalysis(linkjiru::defaultVtsHost, port);

    // Coordination sleeps — wait for connect+auth, then registration.
    static constexpr int vtsConnectDelayMs  = 500;
    static constexpr int vtsRegisterDelayMs = 200;
    std::this_thread::sleep_for(std::chrono::milliseconds(vtsConnectDelayMs));
    processor.requestVtsRegister();
    std::this_thread::sleep_for(std::chrono::milliseconds(vtsRegisterDelayMs));

    const auto blockDuration = std::chrono::duration<double>(static_cast<double>(BLOCK_SIZE) / SAMPLE_RATE);

    std::atomic<bool> writerDone{false};

    /* Near-silence duration before switching to speech signal.
       2.5 s lets the RmsAnalyzer finish calibration first. */
    static constexpr double SILENCE_DURATION_SEC = 2.5;
    static constexpr int SILENCE_BLOCKS          = static_cast<int>((SAMPLE_RATE * SILENCE_DURATION_SEC) / BLOCK_SIZE);

    std::thread writer(
        [&]
        {
            juce::AudioBuffer<float> buffer(1, BLOCK_SIZE);
            juce::MidiBuffer midi;

            for (int block = 0; block < TOTAL_BLOCKS; ++block)
            {
                if (block < SILENCE_BLOCKS)
                {
                    for (int i = 0; i < BLOCK_SIZE; ++i)
                        buffer.setSample(0, i, 0.001f);
                }
                else
                {
                    for (int i = 0; i < BLOCK_SIZE; ++i)
                        buffer.setSample(0, i, 0.5f * std::sin(static_cast<float>(block * BLOCK_SIZE + i) * 0.1f));
                }

                processor.processBlock(buffer, midi);
                std::this_thread::sleep_for(blockDuration);
            }

            writerDone.store(true);
        });

    static constexpr int pollIntervalMs    = 16; // ~60 Hz, matches AnalysisThread
    static constexpr float detectThreshold = 0.5f;

    int nonZeroDetections = 0;
    int totalPolls        = 0;
    while (!writerDone.load())
    {
        const float val = processor.getDetectValue();
        if (val > detectThreshold)
            ++nonZeroDetections;
        ++totalPolls;
        std::this_thread::sleep_for(std::chrono::milliseconds(pollIntervalMs));
    }

    writer.join();

    const bool vtsConnected  = processor.isVtsConnected();
    const bool vtsRegistered = processor.isVtsRegistered();
    const int injections     = mockVts.getInjectCount();

    processor.stopAnalysis();
    processor.releaseResources();
    mockVts.stop();

    EXPECT_TRUE(vtsConnected) << "AnalysisThread never connected to mock VTS on port " << port;

    EXPECT_TRUE(vtsRegistered) << "VTS parameter was never registered";

    EXPECT_GT(injections, 0) << "Mock VTS received 0 InjectParameterDataRequest calls";

    EXPECT_GT(nonZeroDetections, 0) << "Analysis pipeline never detected speech (" << totalPolls << " polls, all 0.0)";

    std::printf("\n  [%s] RealPluginProducerConsumer:\n"
                "    rate=%.0fHz block=%d duration=%ds\n"
                "    blocks: %d total (%d silence + %d signal)\n"
                "    detections: %d/%d polls (%.1f%%)\n"
                "    vts: connected=%s registered=%s injections=%d port=%s\n\n",
                TAG, SAMPLE_RATE, BLOCK_SIZE, TEST_DURATION_SECONDS, TOTAL_BLOCKS, SILENCE_BLOCKS,
                TOTAL_BLOCKS - SILENCE_BLOCKS, nonZeroDetections, totalPolls,
                totalPolls > 0 ? 100.0 * nonZeroDetections / totalPolls : 0.0, vtsConnected ? "yes" : "no",
                vtsRegistered ? "yes" : "no", injections, port.c_str());
}

// Boundary abuse

TEST(RingBuffer_Boundary, WriteLargerThanCapacity)
{
    SharedRingBuffer<16> buf;

    float src[20];
    for (int i = 0; i < 20; ++i)
        src[i] = static_cast<float>(i);

    buf.write(src, 20);

    EXPECT_EQ(buf.getWriteCount(), 20u);

    float dest[16] = {};
    buf.readLastN(dest, 16);

    for (int i = 0; i < 16; ++i)
        EXPECT_FLOAT_EQ(dest[i], static_cast<float>(4 + i));

    std::printf("  [%s] WriteLargerThanCapacity: wrote 20 into cap-16, "
                "writeCount=%llu, last 16 = [%.0f..%.0f]\n",
                TAG, static_cast<unsigned long long>(buf.getWriteCount()), dest[0], dest[15]);
}

/* Snapshot coherence: monotonicity of incrementing counter values is used as a
   proxy for snapshot coherence. If readLastN returns a window where sample[i] <
   sample[i-1], the snapshot straddled a write boundary and mixed old/new data.
   Counter uses modulo to avoid int overflow on long runs. */

TEST(RingBuffer_Concurrent, ReadLastNCoherence)
{
    // Production-sized buffer and window for realistic memory layout.
    static constexpr int CAP    = linkjiru::ringBufferCapacity;
    static constexpr int WINDOW = linkjiru::defaultAnalysisWindow;
    SharedRingBuffer<CAP> buf;

    // High write pressure to maximize chance of catching a torn read.
    static constexpr double SAMPLE_RATE = 192000.0;
    static constexpr int WRITE_BLOCK    = 32;

    /* Counter wraps at 10 M to keep float precision safe
       (~52 s at 192 kHz, well beyond this test's runtime). */
    static constexpr int COUNTER_MOD = 10000000;

    const auto blockDuration = std::chrono::duration<double>(static_cast<double>(WRITE_BLOCK) / SAMPLE_RATE);

    std::atomic<bool> stop{false};
    std::atomic<bool> failed{false};
    std::atomic<int> checksCompleted{0};

    std::thread writer(
        [&]
        {
            float block[WRITE_BLOCK];
            int counter = 0;

            while (!stop.load())
            {
                for (int i = 0; i < WRITE_BLOCK; ++i)
                {
                    block[i] = static_cast<float>(counter % COUNTER_MOD);
                    ++counter;
                }
                buf.write(block, WRITE_BLOCK);
                std::this_thread::sleep_for(blockDuration);
            }
        });

    // Let the writer seed the buffer before we start reading.
    static constexpr int writerWarmupMs  = 100;
    static constexpr int coherenceChecks = 10000;

    std::this_thread::sleep_for(std::chrono::milliseconds(writerWarmupMs));

    float dest[WINDOW];
    for (int iter = 0; iter < coherenceChecks; ++iter)
    {
        buf.readLastN(dest, WINDOW);

        for (int i = 1; i < WINDOW; ++i)
        {
            // A drop is only a torn read if it isn't a counter wrap.
            const bool dropped     = dest[i] < dest[i - 1];
            const bool counterWrap = dest[i - 1] >= static_cast<float>(COUNTER_MOD - WRITE_BLOCK) &&
                                     dest[i] < static_cast<float>(WRITE_BLOCK);
            if (dropped && !counterWrap)
            {
                failed.store(true);
                break;
            }
        }

        if (failed.load())
            break;

        checksCompleted.fetch_add(1);
    }

    stop.store(true);
    writer.join();

    EXPECT_FALSE(failed.load()) << "readLastN returned a non-monotonic window during concurrent writing";

    std::printf("  [%s] ReadLastNCoherence: %d monotonicity checks at 192kHz\n", TAG, checksCompleted.load());
}
