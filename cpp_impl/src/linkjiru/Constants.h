#pragma once

/* ═══════════════════════════════════════════════════════════════════
   Constants.h — Shared compile-time constants for the Linkjiru plugin.

   Values that appear in more than one translation unit live here so
   there is exactly one place to change them. File-local constants
   (UI layout, test parameters, etc.) stay in their respective files.
   ═══════════════════════════════════════════════════════════════════ */

namespace linkjiru
{

// ── Audio pipeline ──────────────────────────────────────────────

/* Ring buffer capacity in samples.  Must be a power of two.
   131072 ≈ 3 s at 44.1 kHz mono, ~0.7 s at 192 kHz.
   Just needs to be big enough to handle the worst-case scenario. */
inline constexpr int ringBufferCapacity = 131072;

/* Default analysis window size in samples.
   Arbitrary value chosen for demonstration purposes. */
inline constexpr int defaultAnalysisWindow = 2048;

// ── VTube Studio ────────────────────────────────────────────────

/* VTS parameter name for speech detection.
   Changing this means re-registering in VTube Studio. */
inline constexpr const char* detectParamName = "LinkjiruDetectLowji";

/* Default VTS WebSocket connection.
   VTube Studio listens on localhost:8001 out of the box. */
inline constexpr const char* defaultVtsHost = "localhost";
inline constexpr const char* defaultVtsPort = "8001";

/* VTS protocol envelope fields — every request/response
   carries these verbatim. */
inline constexpr const char* vtsApiName    = "VTubeStudioPublicAPI";
inline constexpr const char* vtsApiVersion = "1.0";

// ── Plugin identity ─────────────────────────────────────────────

/* Plugin and developer identity.  Used for VTS auth, the editor
   title, and the token storage folder under %APPDATA%.
   Keep in sync with CMake PRODUCT_NAME / COMPANY_NAME. */
inline constexpr const char* pluginName    = "Linkjiru";
inline constexpr const char* developerName = "tomobaji";

// ── Threading ───────────────────────────────────────────────────

/* Thread::stopThread() timeout (ms).  3 s is generous — the
   analysis loop normally exits within one poll interval. */
inline constexpr int threadStopTimeoutMs = 3000;

} // namespace linkjiru
