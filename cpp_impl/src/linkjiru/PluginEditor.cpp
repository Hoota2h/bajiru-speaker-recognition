#include "PluginEditor.h"
#include "Constants.h"

LinkjiruEditor::LinkjiruEditor(LinkjiruProcessor& p) : AudioProcessorEditor(&p), processor(p)
{
    startButton.onClick = [this]
    {
        processor.startAnalysis();
        updateStatus();
    };

    stopButton.onClick = [this] { stopAsync(); };

    restartButton.onClick = [this] { restartAsync(); };

    vtsRegisterButton.onClick = [this] { processor.requestVtsRegister(); };

    statusLabel.setJustificationType(juce::Justification::centred);
    statusLabel.setFont(juce::FontOptions(14.0f));

    detectLabel.setJustificationType(juce::Justification::centred);
    detectLabel.setFont(juce::FontOptions(13.0f));
    detectLabel.setText(juce::String(linkjiru::detectParamName) + " = --", juce::dontSendNotification);
    detectLabel.setColour(juce::Label::textColourId, juce::Colours::grey);

    vtsStatusLabel.setJustificationType(juce::Justification::centred);
    vtsStatusLabel.setFont(juce::FontOptions(12.0f));

    addAndMakeVisible(startButton);
    addAndMakeVisible(stopButton);
    addAndMakeVisible(restartButton);
    addAndMakeVisible(statusLabel);
    addAndMakeVisible(detectLabel);
    addAndMakeVisible(vtsRegisterButton);
    addAndMakeVisible(vtsStatusLabel);

    updateStatus();
    updateVtsStatus();

    startTimerHz(2);
    setSize(400, 420);
}

LinkjiruEditor::~LinkjiruEditor()
{
    stopTimer();
}

void LinkjiruEditor::stopAsync()
{
    if (stoppingInProgress.load())
    {
        return;
    }

    stoppingInProgress.store(true);
    stopButton.setEnabled(false);
    restartButton.setEnabled(false);
    statusLabel.setText("Status: Stopping...", juce::dontSendNotification);
    statusLabel.setColour(juce::Label::textColourId, juce::Colours::orange);

    /* Paranoia: SafePointer guards against use-after-free: if the DAW destroys
       the editor while stopAnalysis() is blocking, the lambda skips
       the write to stoppingInProgress instead of hitting freed memory. */
    auto safeThis = juce::Component::SafePointer<LinkjiruEditor>(this);
    juce::Thread::launch(
        [safeThis, &proc = processor]
        {
            proc.stopAnalysis();
            if (safeThis != nullptr)
            {
                safeThis->stoppingInProgress.store(false);
            }
        });
}

void LinkjiruEditor::restartAsync()
{
    if (stoppingInProgress.load())
    {
        return;
    }

    stoppingInProgress.store(true);
    stopButton.setEnabled(false);
    restartButton.setEnabled(false);
    statusLabel.setText("Status: Restarting...", juce::dontSendNotification);
    statusLabel.setColour(juce::Label::textColourId, juce::Colours::orange);

    auto safeThis = juce::Component::SafePointer<LinkjiruEditor>(this);
    juce::Thread::launch(
        [safeThis, &proc = processor]
        {
            proc.restartAnalysis();
            if (safeThis != nullptr)
            {
                safeThis->stoppingInProgress.store(false);
            }
        });
}

void LinkjiruEditor::timerCallback()
{
    updateVtsStatus();

    if (stoppingInProgress.load())
    {
        return;
    }

    if (!processor.isAnalysisRunning() && !stopButton.isEnabled())
    {
        stopButton.setEnabled(true);
        restartButton.setEnabled(true);
        updateStatus();
    }

    if (processor.isAnalysisRunning())
    {
        const float val = processor.getDetectValue();
        detectLabel.setText(juce::String(linkjiru::detectParamName) + " = " + juce::String(val, 1),
                            juce::dontSendNotification);
        detectLabel.setColour(juce::Label::textColourId, val > 0.5f ? juce::Colours::limegreen : juce::Colours::grey);
    }
    else
    {
        detectLabel.setText(juce::String(linkjiru::detectParamName) + " = --", juce::dontSendNotification);
        detectLabel.setColour(juce::Label::textColourId, juce::Colours::grey);
    }
}

void LinkjiruEditor::paint(juce::Graphics& g)
{
    g.fillAll(juce::Colour(0xff1a1a2e));

    g.setColour(juce::Colours::white);
    g.setFont(juce::Font(juce::FontOptions(22.0f).withStyle("Bold")));
    g.drawText(linkjiru::pluginName, getLocalBounds().removeFromTop(50), juce::Justification::centred);
}

void LinkjiruEditor::resized()
{
    auto area = getLocalBounds().reduced(30);
    area.removeFromTop(60);

    startButton.setBounds(area.removeFromTop(40));
    area.removeFromTop(10);
    stopButton.setBounds(area.removeFromTop(40));
    area.removeFromTop(10);
    restartButton.setBounds(area.removeFromTop(40));
    area.removeFromTop(12);
    statusLabel.setBounds(area.removeFromTop(22));
    area.removeFromTop(4);
    detectLabel.setBounds(area.removeFromTop(22));
    area.removeFromTop(12);
    vtsRegisterButton.setBounds(area.removeFromTop(40));
    area.removeFromTop(8);
    vtsStatusLabel.setBounds(area.removeFromTop(20));
}

void LinkjiruEditor::updateStatus()
{
    if (processor.isAnalysisRunning())
    {
        statusLabel.setText("Status: Running", juce::dontSendNotification);
        statusLabel.setColour(juce::Label::textColourId, juce::Colours::limegreen);
    }
    else
    {
        statusLabel.setText("Status: Stopped", juce::dontSendNotification);
        statusLabel.setColour(juce::Label::textColourId, juce::Colours::grey);
    }
}

void LinkjiruEditor::updateVtsStatus()
{
    const bool running        = processor.isAnalysisRunning();
    const bool connected      = processor.isVtsConnected();
    const bool registered     = processor.isVtsRegistered();
    const bool registerFailed = processor.isVtsRegisterFailed();

    if (!running)
    {
        vtsRegisterButton.setEnabled(false);
        vtsRegisterButton.setButtonText("Register in VTS");
        vtsStatusLabel.setText("VTS: start analysis first", juce::dontSendNotification);
        vtsStatusLabel.setColour(juce::Label::textColourId, juce::Colours::grey);
    }
    else if (registered)
    {
        vtsRegisterButton.setEnabled(false);
        vtsRegisterButton.setButtonText("Registered");
        vtsStatusLabel.setText("VTS: parameter active", juce::dontSendNotification);
        vtsStatusLabel.setColour(juce::Label::textColourId, juce::Colours::limegreen);
    }
    else if (registerFailed && connected)
    {
        vtsRegisterButton.setEnabled(true);
        vtsRegisterButton.setButtonText("Retry Register");
        vtsStatusLabel.setText("VTS: registration failed, try again", juce::dontSendNotification);
        vtsStatusLabel.setColour(juce::Label::textColourId, juce::Colours::red);
    }
    else if (connected)
    {
        vtsRegisterButton.setEnabled(true);
        vtsRegisterButton.setButtonText("Register in VTS");
        vtsStatusLabel.setText("VTS: connected, ready to register", juce::dontSendNotification);
        vtsStatusLabel.setColour(juce::Label::textColourId, juce::Colours::yellow);
    }
    else
    {
        vtsRegisterButton.setEnabled(false);
        vtsRegisterButton.setButtonText("Register in VTS");
        vtsStatusLabel.setText("VTS: waiting for connection...", juce::dontSendNotification);
        vtsStatusLabel.setColour(juce::Label::textColourId, juce::Colours::orange);
    }
}
