#pragma once

#include "PluginProcessor.h"

class LinkjiruEditor final : public juce::AudioProcessorEditor, private juce::Timer
{
public:
    explicit LinkjiruEditor(LinkjiruProcessor&);
    ~LinkjiruEditor() override;

    void paint(juce::Graphics&) override;
    void resized() override;

private:
    LinkjiruProcessor& processor;

    juce::TextButton startButton{"Start Analysis"};
    juce::TextButton stopButton{"Stop Analysis"};
    juce::TextButton restartButton{"Restart Analysis"};
    juce::TextButton vtsRegisterButton{"Register in VTS"};
    juce::Label statusLabel;
    juce::Label detectLabel;
    juce::Label vtsStatusLabel;

    std::atomic<bool> stoppingInProgress{false};

    void timerCallback() override;
    void updateStatus();
    void updateVtsStatus();
    void stopAsync();
    void restartAsync();

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(LinkjiruEditor)
};
