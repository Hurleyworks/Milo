
#pragma once

#include <qms_core/qms_core.h>

namespace Config
{
    // Default socket server port
    constexpr int DEFAULT_SOCKET_PORT = 9875;

    // Buffer sizes
    constexpr size_t SOCKET_BUFFER_SIZE = 4096;

    // Timing
    constexpr int SOCKET_TIMER_MS = 10;
    constexpr int THREAD_SLEEP_MS = 1;
} // namespace Config

// Use the existing moody queue
using CommandQueue = moodycamel::ConcurrentQueue<std::string>;

// Client data structure to track per-client information
struct ClientData
{
    std::string residualBuffer; // Store incomplete messages
    bool sendInProgress = false;
    std::string sendBuffer; // For handling partial sends
};

// Simple socket server implementation that handles client connections
// and message passing
class SocketServerImpl
{
 public:
    SocketServerImpl();
    ~SocketServerImpl();

    // Initialize the socket server on the given port
    bool init (int port);

    // Check for and accept new client connections
    bool acceptNewConnections();

    // Check all clients for incoming data
    void checkAllClients();

    // Get the next message from the moody queue
    std::string getNextMoodyMessage()
    {
        std::string msg;
        bool success = moodyMessages.try_dequeue (msg);
        return success ? msg : "";
    }

    // Send response to all clients with retry for non-blocking sockets
    void sendResponseToAllClients (const std::string& response);

    // Clean up and shut down the socket server
    void shutdown();

    // Get all client sockets (for compatibility)
    const std::vector<SocketHandle>& getClientSockets() const { return clientSockets; }

 private:
    // Socket handles
    SocketHandle serverSocket;

    // Client socket list
    std::vector<SocketHandle> clientSockets;

    // Client data mapping
    std::map<SocketHandle, ClientData> clientDataMap;

    // Running state flag
    bool running;

    // Lock-free queue for messages
    CommandQueue moodyMessages;

    // Check for incoming data on a specific client socket
    // Returns: true if data was received and processed, false otherwise
    // Additionally returns whether the socket was disconnected via outParam
    bool checkClientData (SocketHandle clientSocket, bool& disconnected);

    // Process received data from a client
    void processClientData (SocketHandle clientSocket, const std::string& data);

    // Try to send data to a client, handling partial sends
    bool trySendToClient (SocketHandle clientSocket, const std::string& message);

    // Check and continue pending sends
    void checkPendingSends();
};