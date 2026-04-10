#include <juce_gui_basics/juce_gui_basics.h>
#include <gtest/gtest.h>

int main(int argc, char** argv)
{
    juce::ScopedJuceInitialiser_GUI gui;
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
