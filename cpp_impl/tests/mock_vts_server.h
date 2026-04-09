#pragma once

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#include <boost/beast/core.hpp>
#include <boost/beast/websocket.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/asio/steady_timer.hpp>
#include <nlohmann/json.hpp>

#include "Constants.h"

#include <atomic>
#include <string>
#include <thread>

/* Fake VTube Studio Emulation(?) that does the bare minimum to keep our client happy.
   Uses synchronous I/O because Boost.Asio's async API is a template
   hellscape I refuse to drag into test code. */
class MockVtsServer
{
    using beast_ws = boost::beast::websocket::stream<boost::beast::tcp_stream>;
    using tcp      = boost::asio::ip::tcp;
    using json     = nlohmann::json;
    using flat_buf = boost::beast::flat_buffer;

public:
    explicit MockVtsServer(const unsigned short port = 0) : acceptor(ioc, tcp::endpoint(tcp::v4(), port))
    {
        actualPort = acceptor.local_endpoint().port();
    }

    ~MockVtsServer() { stop(); }

    unsigned short getPort() const { return actualPort; }

    void start()
    {
        serverThread = std::thread([this] { run(); });
    }

    void stop()
    {
        // Close acceptor first so the poll loop wakes immediately
        // instead of sleeping for another pollIntervalMs.
        boost::system::error_code ec;
        acceptor.close(ec);

        running.store(false);

        if (serverThread.joinable())
        {
            serverThread.join();
        }
    }

    int getInjectCount() const { return injectCount.load(); }

private:
    static constexpr int pollIntervalMs = 50;  // non-blocking accept poll (ms)
    static constexpr int wsTimeoutMs    = 500; // WS read/write timeout (ms)

    /* Fake auth token.  The client encrypts it with DPAPI,
       so the value itself is irrelevant. */
    static constexpr const char* mockAuthToken = "mock-test-token";

    boost::asio::io_context ioc;
    tcp::acceptor acceptor;
    unsigned short actualPort = 0;
    std::atomic<bool> running{true};
    std::atomic<int> injectCount{0};
    std::thread serverThread;

    void run()
    {
        while (running.load())
        {
            try
            {
                /* Polling accept because blocking accept() on Windows
                   cannot be reliably interrupted from another thread.
                   Thanks Microslop. */
                acceptor.non_blocking(true);

                tcp::socket socket(ioc);
                boost::system::error_code ec;

                while (running.load())
                {
                    acceptor.accept(socket, ec);
                    if (!ec)
                    {
                        break;
                    }

                    if (ec == boost::asio::error::would_block || ec == boost::asio::error::try_again)
                    {
                        std::this_thread::sleep_for(std::chrono::milliseconds(pollIntervalMs));
                        continue;
                    }

                    // Stuff happens, probably acceptor was closed to shut down the server.
                    return;
                }

                if (!running.load())
                {
                    return;
                }

                beast_ws ws(std::move(socket));
                ws.accept();

                // Timeout so we can actually shut down. Beast has no cancellation token.
                boost::beast::get_lowest_layer(ws).expires_after(std::chrono::milliseconds(wsTimeoutMs));

                while (running.load())
                {
                    flat_buf buf;
                    boost::system::error_code readEc;
                    ws.read(buf, readEc);

                    if (readEc == boost::beast::error::timeout)
                    {
                        // Timed out, not an error. Re-arm and spin again.
                        boost::beast::get_lowest_layer(ws).expires_after(std::chrono::milliseconds(wsTimeoutMs));
                        continue;
                    }

                    if (readEc)
                    {
                        break; // client bailed or something broke. Either way, not our problem.
                    }

                    const auto request = json::parse(boost::beast::buffers_to_string(buf.data()));

                    const auto messageType = request.value("messageType", "");
                    const auto requestID   = request.value("requestID", "");

                    json response = makeResponse(messageType, requestID);

                    boost::beast::get_lowest_layer(ws).expires_after(std::chrono::milliseconds(wsTimeoutMs));
                    ws.write(boost::asio::buffer(response.dump()));
                }
            }
            catch (...)
            {
                if (!running.load())
                {
                    return;
                }
            }
        }
    }

    json makeResponse(const std::string& messageType, const std::string& requestID)
    {
        json data;

        /* Basically copied from our client's code.
           Just enough to keep the client from freaking out and disconnecting. */

        if (messageType == "AuthenticationTokenRequest")
        {
            data = {{"authenticationToken", mockAuthToken}};
            return envelope("AuthenticationTokenResponse", data, requestID);
        }

        if (messageType == "AuthenticationRequest")
        {
            data = {{"authenticated", true}, {"reason", ""}};
            return envelope("AuthenticationResponse", data, requestID);
        }

        if (messageType == "ParameterCreationRequest")
        {
            data = {};
            return envelope("ParameterCreationResponse", data, requestID);
        }

        if (messageType == "InjectParameterDataRequest")
        {
            injectCount.fetch_add(1);
            data = {};
            return envelope("InjectParameterDataResponse", data, requestID);
        }

        data = {{"errorID", 1}, {"message", "unknown request type"}};
        return envelope("APIError", data, requestID);
    }

    static json envelope(const std::string& messageType, const json& data, const std::string& requestID)
    {
        return {{"apiName", linkjiru::vtsApiName},
                {"apiVersion", linkjiru::vtsApiVersion},
                {"requestID", requestID},
                {"messageType", messageType},
                {"data", data}};
    }
};
