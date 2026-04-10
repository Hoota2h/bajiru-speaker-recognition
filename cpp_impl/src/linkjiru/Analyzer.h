#pragma once

class Analyzer
{
    /* Abstract base class for analysis model
       You can implement this with any ML model you like, or even just a simple energy threshold.
       Currently I just set this up as a C++ interface to avoid adding dependencies to the plugin processor, and
       as a clean way to separate the analysis code from the plugin code. The AnalysisThread owns an Analyzer and calls
       its methods to do the actual work. */
public:
    virtual ~Analyzer() = default;

    /* Feed audio samples during calibration phase.
       Keep calling until isCalibrated() returns true.
       Yes, you have to poll. No, there's no callback. */
    virtual void calibrate(const float* data, int numSamples) = 0;

    [[nodiscard]] virtual bool isCalibrated() const                                    = 0;
    [[nodiscard]] virtual bool isSpeechActive(const float* data, int numSamples) const = 0;
    [[nodiscard]] virtual float getThreshold() const                                   = 0;
};
