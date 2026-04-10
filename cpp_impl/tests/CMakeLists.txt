include(FetchContent)
FetchContent_Declare(googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.15.2
    GIT_SHALLOW    TRUE)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Benchmarks — pure C++ only, no JUCE dependency
add_executable(LinkjiruBenchmarks
    bench_ring_buffer.cpp
    bench_chain_latency.cpp)

target_include_directories(LinkjiruBenchmarks
    PRIVATE ${CMAKE_SOURCE_DIR}/src/linkjiru)

target_compile_definitions(LinkjiruBenchmarks
    PRIVATE
        _WIN32_WINNT=0x0A00
        _CRT_SECURE_NO_WARNINGS)

target_link_libraries(LinkjiruBenchmarks
    PRIVATE GTest::gtest_main)

# Unit tests — links SharedCode for full JUCE plugin access.
# Uses a custom main() that initialises the JUCE MessageManager
# via ScopedJuceInitialiser_GUI so we can instantiate the real
# PluginProcessor in tests.
add_executable(LinkjiruTests
    gtest_main.cpp
    test_shared_ring_buffer.cpp)

target_link_libraries(LinkjiruTests
    PRIVATE
        SharedCode
        Boost::beast
        nlohmann_json::nlohmann_json
        GTest::gtest)

# JUCE plugin definitions (JucePlugin_Name, format flags, etc.) are generated
# by juce_add_plugin and only exist on the Linkjiru target. We need them here
# because PluginProcessor.cpp references JucePlugin_Name. The generator
# expression forwards them so tests compile the same code as the plugin.
# JUCE_MODAL_LOOPS_PERMITTED is test-only (needed for ScopedJuceInitialiser_GUI).
target_compile_definitions(LinkjiruTests
    PRIVATE
        JUCE_MODAL_LOOPS_PERMITTED=1
        _WIN32_WINNT=0x0A00
        _CRT_SECURE_NO_WARNINGS
        $<TARGET_PROPERTY:Linkjiru,COMPILE_DEFINITIONS>)
