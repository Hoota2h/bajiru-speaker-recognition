#pragma once

#include "Analyzer.h"
#include <cmath>
#include <algorithm>

class RmsAnalyzer final : public Analyzer
{
public:
    struct Config
    {
        float noiseFloorMultiplier = 4.0f;   // threshold = noiseFloor * multiplier
        float minThreshold         = 0.005f; // absolute floor if calibration is near-silent
        int calibrationSamples     = 44100;  // ~1s at 44.1kHz (overridden by AnalysisThread)
    };

    RmsAnalyzer() : config(), currentThreshold(config.minThreshold) {}
    explicit RmsAnalyzer(const Config& cfg) : config(cfg), currentThreshold(cfg.minThreshold) {}

    static float computeRms(const float* data, const int numSamples)
    {
        if (numSamples <= 0)
            return 0.0f;

        float sumSq = 0.0f;
        for (int i = 0; i < numSamples; ++i)
            sumSq += data[i] * data[i];

        return std::sqrt(sumSq / static_cast<float>(numSamples));
    }

    /* Feed audio samples during calibration phase.
       Call repeatedly until isCalibrated() returns true. */
    void calibrate(const float* data, const int numSamples) override
    {
        if (calibrated)
            return;

        for (int i = 0; i < numSamples && calibrationCount < config.calibrationSamples; ++i)
        {
            calibrationSumSq += static_cast<double>(data[i]) * data[i];
            ++calibrationCount;
        }

        if (calibrationCount >= config.calibrationSamples)
        {
            const auto noiseFloorRms =
                static_cast<float>(std::sqrt(calibrationSumSq / static_cast<double>(calibrationCount)));
            currentThreshold = std::max(config.minThreshold, noiseFloorRms * config.noiseFloorMultiplier);
            calibrated       = true;
        }
    }

    bool isSpeechActive(const float* data, const int numSamples) const override
    {
        return computeRms(data, numSamples) > currentThreshold;
    }

    [[nodiscard]] float getThreshold() const override { return currentThreshold; }
    [[nodiscard]] bool isCalibrated() const override { return calibrated; }

private:
    Config config;
    float currentThreshold;
    bool calibrated         = false;
    int calibrationCount    = 0;
    double calibrationSumSq = 0.0;
};
