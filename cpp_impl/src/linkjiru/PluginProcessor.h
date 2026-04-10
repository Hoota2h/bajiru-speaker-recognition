#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include "Constants.h"
#include "SharedRingBuffer.h"
#include <string>
#include <vector>

class AnalysisThread;

class LinkjiruProcessor final : public juce::AudioProcessor
{
public:
    LinkjiruProcessor();
    ~LinkjiruProcessor() override;

    void prepareToPlay(double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;
    bool isBusesLayoutSupported(const BusesLayout& layouts) const override;
    void processBlock(juce::AudioBuffer<float>&, juce::MidiBuffer&) override;

    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override;

    const juce::String getName() const override;
    bool acceptsMidi() const override;
    bool producesMidi() const override;
    bool isMidiEffect() const override;
    double getTailLengthSeconds() const override;

    int getNumPrograms() override;
    int getCurrentProgram() override;
    void setCurrentProgram(int index) override;
    const juce::String getProgramName(int index) override;
    void changeProgramName(int index, const juce::String& newName) override;

    void getStateInformation(juce::MemoryBlock& destData) override;
    void setStateInformation(const void* data, int sizeInBytes) override;

    void startAnalysis(const std::string& vtsHost = linkjiru::defaultVtsHost,
                       const std::string& vtsPort = linkjiru::defaultVtsPort);
    void stopAnalysis();
    void restartAnalysis();
    bool isAnalysisRunning() const { return analysisRunning.load(); }

    // VTS state passthrough — delegates to AnalysisThread atomics
    bool isVtsConnected() const;
    bool isVtsRegistered() const;
    bool isVtsRegisterFailed() const;
    void requestVtsRegister() const;
    float getDetectValue() const;

private:
    SharedRingBuffer<linkjiru::ringBufferCapacity> sharedBuffer;

    std::atomic<bool> analysisRunning{false};
    std::unique_ptr<AnalysisThread> analysisThread;

    double currentSampleRate = 44100.0;
    int currentBlockSize     = 512;
    std::vector<float> monoMixBuf;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(LinkjiruProcessor)
};
