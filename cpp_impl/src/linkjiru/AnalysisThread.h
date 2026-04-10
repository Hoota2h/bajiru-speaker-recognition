#pragma once

#include <juce_core/juce_core.h>
#include "Constants.h"
#include "SharedRingBuffer.h"
#include "Analyzer.h"
#include "VTubeStudioClient.h"
#include <memory>
#include <string>
#include <vector>

class AnalysisThread final : public juce::Thread
{
public:
    struct Config
    {
        int analysisWindowSamples = linkjiru::defaultAnalysisWindow;
        int pollIntervalMs        = 16;  // ~60 fps to match VTube Studio
        int sustainMs             = 100; // hold speech state after last detection, tunable parameter
        std::string vtsHost       = linkjiru::defaultVtsHost;
        std::string vtsPort       = linkjiru::defaultVtsPort;
    };

    /* Yes, two constructors. We can't merge them with a default
       argument (const Config& cfg = Config{}). C++ refuses to evaluate
       Config's default member initializers inside a default argument
       before the enclosing class is fully defined Smadge
       Also clang-tidy fights it too, thus the change */
    AnalysisThread(const SharedRingBuffer<linkjiru::ringBufferCapacity>& buffer, std::unique_ptr<Analyzer> analyzerIn)
        : Thread("LinkjiruAnalysis"), sharedBuffer(buffer), analyzer(std::move(analyzerIn))
    {
    }

    AnalysisThread(const SharedRingBuffer<linkjiru::ringBufferCapacity>& buffer, std::unique_ptr<Analyzer> analyzerIn,
                   const Config& cfg)
        : Thread("LinkjiruAnalysis"), sharedBuffer(buffer), analyzer(std::move(analyzerIn)), config(cfg)
    {
    }

    ~AnalysisThread() override { stopThread(linkjiru::threadStopTimeoutMs); }

    // VTS state (read by UI via processor passthrough)
    bool isVtsConnected() const { return vtsConnected.load(); }
    bool isVtsRegistered() const { return vtsRegistered.load(); }
    bool isRegisterFailed() const { return registerFailed.load(); }

    // Model output (read by UI via processor passthrough)
    float getDetectValue() const { return detectValue.load(); }

    // Called from UI thread via processor — sets a flag the run loop picks up
    void requestVtsRegister()
    {
        registerFailed.store(false);
        registerRequested.store(true);
    }

    void run() override
    {
        std::vector<float> windowBuf(config.analysisWindowSamples);

        // VTube Studio connection
        VTubeStudioClient vtsClient;
        int64_t lastReconnectAttempt = 0;

        /* How long to wait between VTS reconnection attempts (ms).
           5 seconds avoids hammering the socket when VTS is not running. */
        static constexpr int reconnectIntervalMs = 5000;

        bool currentlySpeaking = false;
        int64_t lastSpeechTime = 0;

        while (!threadShouldExit())
        {
            const auto now = juce::Time::currentTimeMillis();

            // Manage VTS connection
            if (!vtsClient.isConnected())
            {
                vtsConnected.store(false);
                vtsRegistered.store(false);

                if (now - lastReconnectAttempt >= reconnectIntervalMs)
                {
                    lastReconnectAttempt = now;

                    if (vtsClient.connect(config.vtsHost, config.vtsPort) && vtsClient.authenticate())
                    {
                        vtsConnected.store(true);
                    }
                }
            }

            // Handle registration request from UI
            if (vtsClient.isConnected() && !vtsRegistered.load() && registerRequested.load())
            {
                if (vtsClient.registerParameter(linkjiru::detectParamName, "1 when speaker detected, 0 otherwise", 0.0f,
                                                1.0f, 0.0f))
                {
                    vtsRegistered.store(true);
                }
                else
                {
                    registerFailed.store(true);
                }

                // Request consumed regardless of outcome — user can retry via UI.
                registerRequested.store(false);
            }

            // Read latest audio from ring buffer
            sharedBuffer.readLastN(windowBuf.data(), config.analysisWindowSamples);

            // Calibration phase
            if (!analyzer->isCalibrated())
            {
                analyzer->calibrate(windowBuf.data(), config.analysisWindowSamples);
                sleep(config.pollIntervalMs);
                continue;
            }

            // Speech detection
            const bool speechDetected = analyzer->isSpeechActive(windowBuf.data(), config.analysisWindowSamples);

            if (speechDetected)
            {
                lastSpeechTime    = now;
                currentlySpeaking = true;
            }
            else if (currentlySpeaking && (now - lastSpeechTime > config.sustainMs))
            {
                currentlySpeaking = false;
            }

            const float value = currentlySpeaking ? 1.0f : 0.0f;
            detectValue.store(value);

            // Send to VTube Studio every frame (only if registered)
            if (vtsClient.isConnected() && vtsRegistered.load())
            {
                vtsClient.injectParameter(linkjiru::detectParamName, value);
            }

            sleep(config.pollIntervalMs);
        }

        // Clean shutdown
        if (vtsClient.isConnected())
        {
            if (vtsRegistered.load())
            {
                vtsClient.injectParameter(linkjiru::detectParamName, 0.0f);
            }

            vtsClient.disconnect();
        }

        detectValue.store(0.0f);
        vtsConnected.store(false);
        vtsRegistered.store(false);
    }

private:
    const SharedRingBuffer<linkjiru::ringBufferCapacity>& sharedBuffer;
    std::unique_ptr<Analyzer> analyzer;
    Config config;

    std::atomic<bool> vtsConnected{false};
    std::atomic<bool> vtsRegistered{false};
    std::atomic<bool> registerRequested{false};
    std::atomic<bool> registerFailed{false};
    std::atomic<float> detectValue{0.0f};

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(AnalysisThread)
};
