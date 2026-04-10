#ifndef _WIN32
#error "VTubeStudioClient requires Windows (MSVC). DPAPI and Winsock are not available on other platforms."
#endif

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#include "VTubeStudioClient.h"

#include <boost/beast/core.hpp>
#include <boost/beast/websocket.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <nlohmann/json.hpp>

#include <filesystem>
#include <fstream>
#include <vector>
#include <shlobj.h>
#include <wincrypt.h>
#pragma comment(lib, "crypt32.lib")

namespace beast     = boost::beast;
namespace websocket = beast::websocket;
namespace net       = boost::asio;
using tcp           = net::ip::tcp;
using json          = nlohmann::json;

namespace
{

/*  Max inbound WebSocket (Ws/WS) message (bytes).
    256 KB covers the largest VTS responses (model lists, etc.). */
constexpr std::size_t maxWsMessageSize = std::size_t{256} * 1024;

/* Temp-file suffix for atomic write-then-rename in saveToken(). */
constexpr const char* tokenTmpSuffix = ".tmp";

} // namespace

struct VTubeStudioClient::WebSocketState
{
    // ReSharper disable once CppDFATimeOver
    net::io_context ioc;
    std::unique_ptr<websocket::stream<beast::tcp_stream>> ws;

    // ReSharper disable once CppMemberFunctionMayBeConst
    /* Caller MUST hold sendMutex before calling this.
       Cannot be const: ws->write() and ws->read() advance the Beast stream's
       internal state machine. Marking this const would compile (the unique_ptr
       is const, not the stream it points to) but misrepresents the contract —
       if anything downstream enforces const on WebSocketState, the call breaks
       or requires a const_cast that hides real mutation. */
    [[nodiscard]] json sendRequest(const json& request, const int timeoutMs) // NOLINT(*-make-member-function-const)
    {
        std::string payload = request.dump();

        beast::get_lowest_layer(*ws).expires_after(std::chrono::milliseconds(timeoutMs));

        ws->write(net::buffer(payload));

        beast::flat_buffer buf;
        ws->read(buf);

        return json::parse(beast::buffers_to_string(buf.data()));
    }
};

// Helpers

static json makeEnvelope(const std::string& messageType, const json& data,
                         const std::string& requestID = "linkjiru-req")
{
    return {{"apiName", linkjiru::vtsApiName},
            {"apiVersion", linkjiru::vtsApiVersion},
            {"requestID", requestID},
            {"messageType", messageType},
            {"data", data}};
}

// Lifecycle

VTubeStudioClient::VTubeStudioClient() = default;

VTubeStudioClient::~VTubeStudioClient()
{
    disconnect();
}

bool VTubeStudioClient::connect(const std::string& host, const std::string& port)
{
    std::lock_guard<std::mutex> lock(sendMutex);

    if (wsState && wsState->ws)
    {
        try
        {
            if (wsState->ws->is_open())
            {
                beast::get_lowest_layer(*wsState->ws).expires_after(std::chrono::milliseconds(OP_TIMEOUT_MS));
                wsState->ws->close(websocket::close_code::normal);
            }
        }
        catch (...)
        {
        }
    }

    try
    {
        wsState = std::make_unique<WebSocketState>();

        tcp::resolver resolver(wsState->ioc);
        auto const results = resolver.resolve(host, port);

        wsState->ws = std::make_unique<websocket::stream<beast::tcp_stream>>(wsState->ioc);

        beast::get_lowest_layer(*wsState->ws).expires_after(std::chrono::milliseconds(OP_TIMEOUT_MS));
        beast::get_lowest_layer(*wsState->ws).connect(results);

        beast::get_lowest_layer(*wsState->ws).expires_after(std::chrono::milliseconds(OP_TIMEOUT_MS));
        wsState->ws->handshake(host + ":" + port, "/");

        wsState->ws->read_message_max(maxWsMessageSize);

        connected.store(true);
        return true;
    }
    catch (...)
    {
        wsState.reset();
        connected.store(false);
        return false;
    }
}

void VTubeStudioClient::disconnect()
{
    std::lock_guard<std::mutex> lock(sendMutex);

    if (wsState && wsState->ws)
    {
        try
        {
            if (wsState->ws->is_open())
            {
                beast::get_lowest_layer(*wsState->ws).expires_after(std::chrono::milliseconds(OP_TIMEOUT_MS));
                wsState->ws->close(websocket::close_code::normal);
            }
        }
        catch (...)
        {
        }
    }

    wsState.reset();
    connected.store(false);
}

/* Authentication is a two-step process against the VTube Studio API:

   1. Try the stored token first. On previous runs we saved a DPAPI-encrypted
      auth token to %APPDATA%/Linkjiru/vts_token.dat. If it exists and VTS
      accepts it, we're done — no user interaction needed.

   2. If there's no stored token or VTS rejects it (revoked, different VTS
      instance, etc.), we send an AuthenticationTokenRequest. VTS shows a
      popup asking the user to approve the "Linkjiru" plugin. This blocks
      for up to 60 seconds (AUTH_POPUP_TIMEOUT_MS) waiting for the user to
      click allow. On approval, VTS returns a token which we encrypt with
      DPAPI and write atomically (tmp + rename) to vts_token.dat.

   The token is machine- and user-scoped (DPAPI). It survives VTS restarts
   but not Windows user changes or OS reinstalls. If the encrypted file is
   corrupt, loadToken deletes it and falls through to step 2.

   Why encrypt: a VTS auth token grants full API access — parameter injection,
   hotkey triggers, model loading, expression changes. A stolen plaintext
   token lets any local process silently puppet the avatar. VST plugins live
   in shared DAW processes alongside third-party code, and %APPDATA% is
   readable by any process running as the same user. DPAPI ensures only this
   user on this machine can recover the token, which matches VTS's own trust
   boundary (localhost, same session). It costs us nothing at runtime and
   closes the "malicious plugin reads our token file" vector entirely. */

bool VTubeStudioClient::authenticate()
{
    std::lock_guard<std::mutex> lock(sendMutex);

    if (!wsState || !connected.load())
    {
        return false;
    }

    std::string token;
    if (loadToken(token))
    {
        try
        {
            const auto req = makeEnvelope("AuthenticationRequest",
                                          {{"pluginName", linkjiru::pluginName},
                                           {"pluginDeveloper", linkjiru::developerName},
                                           {"authenticationToken", token}},
                                          "auth-stored");

            auto resp = wsState->sendRequest(req, OP_TIMEOUT_MS);

            if (resp.contains("data") && resp["data"].value("authenticated", false))
            {
                return true;
            }
        }
        catch (...)
        {
            wsState.reset();
            connected.store(false);
            return false;
        }
    }

    try
    {
        const auto tokenReq = makeEnvelope(
            "AuthenticationTokenRequest",
            {{"pluginName", linkjiru::pluginName}, {"pluginDeveloper", linkjiru::developerName}}, "auth-token-req");

        auto tokenResp = wsState->sendRequest(tokenReq, AUTH_POPUP_TIMEOUT_MS);

        if (!tokenResp.contains("data") || !tokenResp["data"].contains("authenticationToken"))
        {
            return false;
        }

        token = tokenResp["data"]["authenticationToken"].get<std::string>();
        saveToken(token);

        const auto authReq = makeEnvelope("AuthenticationRequest",
                                          {{"pluginName", linkjiru::pluginName},
                                           {"pluginDeveloper", linkjiru::developerName},
                                           {"authenticationToken", token}},
                                          "auth-req");

        auto authResp = wsState->sendRequest(authReq, OP_TIMEOUT_MS);

        return authResp.contains("data") && authResp["data"].value("authenticated", false);
    }
    catch (...)
    {
        wsState.reset();
        connected.store(false);
        return false;
    }
}

// VTS API Methods

bool VTubeStudioClient::registerParameter(const std::string& paramId, const std::string& explanation, float minValue,
                                          float maxValue, float defaultValue)
{
    std::lock_guard<std::mutex> lock(sendMutex);

    if (!wsState || !connected.load())
    {
        return false;
    }

    try
    {
        const auto req = makeEnvelope("ParameterCreationRequest",
                                      {{"parameterName", paramId},
                                       {"explanation", explanation},
                                       {"min", minValue},
                                       {"max", maxValue},
                                       {"defaultValue", defaultValue}},
                                      "param-create");

        const auto resp = wsState->sendRequest(req, OP_TIMEOUT_MS);
        return resp.value("messageType", "") == "ParameterCreationResponse";
    }
    catch (...)
    {
        wsState.reset();
        connected.store(false);
        return false;
    }
}

bool VTubeStudioClient::injectParameter(const std::string& paramId, float value)
{
    std::lock_guard<std::mutex> lock(sendMutex);

    if (!wsState || !connected.load())
    {
        return false;
    }

    try
    {
        const auto req = makeEnvelope("InjectParameterDataRequest",
                                      {{"faceFound", false},
                                       {"mode", "set"},
                                       {"parameterValues", json::array({{{"id", paramId}, {"value", value}}})}},
                                      "inject-param");

        const auto resp = wsState->sendRequest(req, OP_TIMEOUT_MS);
        return resp.value("messageType", "") == "InjectParameterDataResponse";
    }
    catch (...)
    {
        wsState.reset();
        connected.store(false);
        return false;
    }
}

// Token Persistence (DPAPI encrypted)

std::string VTubeStudioClient::getTokenFilePath()
{
    wchar_t* widePath = nullptr;
    if (SHGetKnownFolderPath(FOLDERID_RoamingAppData, 0, nullptr, &widePath) != S_OK)
    {
        return {};
    }

    const int len = WideCharToMultiByte(CP_UTF8, 0, widePath, -1, nullptr, 0, nullptr, nullptr);
    if (len <= 1)
    {
        CoTaskMemFree(widePath);
        return {};
    }

    std::string result(static_cast<size_t>(len - 1), '\0');
    const int written = WideCharToMultiByte(CP_UTF8, 0, widePath, -1, result.data(), len, nullptr, nullptr);
    CoTaskMemFree(widePath);

    if (written == 0)
    {
        return {};
    }

    return result + "\\" + linkjiru::pluginName + "\\vts_token.dat";
}

bool VTubeStudioClient::loadToken(std::string& token)
{
    const std::string path = getTokenFilePath();
    if (path.empty())
    {
        return false;
    }

    std::ifstream file(path, std::ios::binary);
    if (!file.is_open())
    {
        return false;
    }

    std::string encrypted((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    if (encrypted.empty())
    {
        return false;
    }

    DATA_BLOB encryptedBlob;
    encryptedBlob.pbData = reinterpret_cast<BYTE*>(encrypted.data());
    encryptedBlob.cbData = static_cast<DWORD>(encrypted.size());

    DATA_BLOB decryptedBlob;
    if (!CryptUnprotectData(&encryptedBlob, nullptr, nullptr, nullptr, nullptr, 0, &decryptedBlob))
    {
        std::filesystem::remove(path);
        return false;
    }

    token.assign(reinterpret_cast<char*>(decryptedBlob.pbData), decryptedBlob.cbData);
    LocalFree(decryptedBlob.pbData);

    return !token.empty();
}

bool VTubeStudioClient::saveToken(const std::string& token)
{
    const std::string path = getTokenFilePath();
    if (path.empty())
    {
        return false;
    }

    std::filesystem::create_directories(std::filesystem::path(path).parent_path());

    std::vector<BYTE> plainCopy(token.begin(), token.end());
    DATA_BLOB plainBlob;
    plainBlob.pbData = plainCopy.data();
    plainBlob.cbData = static_cast<DWORD>(plainCopy.size());

    DATA_BLOB encryptedBlob;
    if (!CryptProtectData(&plainBlob, nullptr, nullptr, nullptr, nullptr, 0, &encryptedBlob))
    {
        return false;
    }

    const std::string tmpPath = path + tokenTmpSuffix;
    {
        std::ofstream file(tmpPath, std::ios::binary);
        if (!file.is_open())
        {
            LocalFree(encryptedBlob.pbData);
            return false;
        }

        file.write(reinterpret_cast<char*>(encryptedBlob.pbData), static_cast<std::streamsize>(encryptedBlob.cbData));
        LocalFree(encryptedBlob.pbData);

        if (!file.good())
        {
            std::filesystem::remove(tmpPath);
            return false;
        }
    }

    std::error_code ec;
    std::filesystem::rename(tmpPath, path, ec);
    if (ec)
    {
        std::filesystem::remove(tmpPath);
        return false;
    }

    return true;
}
