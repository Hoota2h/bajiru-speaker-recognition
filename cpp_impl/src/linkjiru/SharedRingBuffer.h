#pragma once

#include <array>
#include <atomic>
#include <algorithm>
#include <cstdint>

template <int Capacity> class SharedRingBuffer
{
    static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be a power of two");
    static constexpr uint64_t Mask = static_cast<uint64_t>(Capacity) - 1;

public:
    /* Write samples into the ring buffer. Real-time safe (wait-free).
       Only one thread may call write() (the audio thread).
       numSamples may exceed Capacity — the bitmask wraps indices safely,
       only the last Capacity samples are retained in the buffer, and
       writeCount advances by the full numSamples. */
    void write(const float* data, const int numSamples)
    {
        const auto pos = writeCount.load(std::memory_order_relaxed);
        for (int i = 0; i < numSamples; ++i)
            buffer[static_cast<std::size_t>((pos + i) & Mask)] = data[i];
        writeCount.store(pos + numSamples, std::memory_order_release);
    }

    /* Snapshot the most recent N samples into dest. Thread-safe for any reader.
       n must be <= Capacity; clamped if larger. */
    void readLastN(float* dest, int n) const
    {
        if (n > Capacity)
            n = Capacity;

        const auto currentWrite = writeCount.load(std::memory_order_acquire);

        if (currentWrite == 0)
        {
            std::fill_n(dest, n, 0.0f);
            return;
        }

        int toRead;
        uint64_t start;

        if (currentWrite >= static_cast<uint64_t>(n))
        {
            start  = currentWrite - n;
            toRead = n;
        }
        else
        {
            start  = 0;
            toRead = static_cast<int>(currentWrite);
            std::fill_n(dest, n - toRead, 0.0f);
            dest += (n - toRead);
        }

        for (int i = 0; i < toRead; ++i)
            dest[i] = buffer[static_cast<std::size_t>((start + i) & Mask)];
    }

    struct ReadResult
    {
        int samplesRead = 0;
        bool overrun    = false;
    };

    /* Sequential read using a caller-managed cursor. Thread-safe for any reader.
       Each reader maintains its own cursor independently. */
    ReadResult read(float* dest, const int maxSamples, uint64_t& cursor) const
    {
        const auto currentWrite = writeCount.load(std::memory_order_acquire);

        bool overrun = false;
        if (currentWrite > cursor + static_cast<uint64_t>(Capacity))
        {
            cursor  = currentWrite - static_cast<uint64_t>(Capacity);
            overrun = true;
        }

        const auto available = currentWrite - cursor;
        const int toRead     = static_cast<int>(std::min(available, static_cast<uint64_t>(maxSamples)));

        for (int i = 0; i < toRead; ++i)
            dest[i] = buffer[static_cast<std::size_t>((cursor + i) & Mask)];

        cursor += toRead;
        return {toRead, overrun};
    }

    uint64_t getWriteCount() const { return writeCount.load(std::memory_order_acquire); }

private:
    std::array<float, Capacity> buffer{};

    /* Monotonically increasing total samples written. At 192kHz this wraps
       after ~3 trillion years. If it ever wraps, reader cursors and the
       overrun check in read() would produce incorrect results. Not fixable
       without a wider counter, but not a practical concern. */
    std::atomic<uint64_t> writeCount{0};
};
