#include "SocketServerImpl.h"

SocketServerImpl::SocketServerImpl() :
    serverSocket (INVALID_SOCKET_HANDLE),
    running (false)
{
    // Initialize platform-specific socket library if needed
#ifdef _WIN32
    // Initialize Winsock
    WSADATA wsaData;
    if (WSAStartup (MAKEWORD (2, 2), &wsaData) != 0)
    {
        LOG (WARNING) << "WSAStartup failed";
        return;
    }
#endif
}

SocketServerImpl::~SocketServerImpl()
{
    shutdown();

#ifdef _WIN32
    // Clean up Winsock
    WSACleanup();
#endif
}

bool SocketServerImpl::init (int port)
{
    LOG (DBUG) << "SocketServerImpl::init called with port " << port;

    // Create a socket
    serverSocket = socket (AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (serverSocket == INVALID_SOCKET_HANDLE)
    {
#ifdef _WIN32
        LOG (WARNING) << "Socket creation failed: " << WSAGetLastError();
#else
        LOG (WARNING) << "Socket creation failed: " << strerror (errno);
#endif
        return false;
    }

    // Set socket to non-blocking mode - CRITICAL for LightWave's single-threaded environment
#ifdef _WIN32
    u_long mode = 1; // 1 = non-blocking
    if (ioctlsocket (serverSocket, FIONBIO, &mode) != 0)
    {
        CloseSocketHandle (serverSocket);
        serverSocket = INVALID_SOCKET_HANDLE;
        return false;
    }
#else
    int flags = fcntl (serverSocket, F_GETFL, 0);
    if (flags == -1 || fcntl (serverSocket, F_SETFL, flags | O_NONBLOCK) == -1)
    {
        CloseSocketHandle (serverSocket);
        serverSocket = INVALID_SOCKET_HANDLE;
        return false;
    }
#endif

    // Enable address reuse to prevent "address already in use" errors
    int opt = 1;
    setsockopt (serverSocket, SOL_SOCKET, SO_REUSEADDR, (const char*)&opt, sizeof (opt));

    // Set up address structure
    struct sockaddr_in serverAddr;
    memset (&serverAddr, 0, sizeof (serverAddr));
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_addr.s_addr = INADDR_ANY;
    serverAddr.sin_port = htons (static_cast<unsigned short> (port));

    // Bind socket
    if (bind (serverSocket, (struct sockaddr*)&serverAddr, sizeof (serverAddr)) == SOCKET_ERROR_HANDLE)
    {
#ifdef _WIN32
        LOG (WARNING) << "Bind failed: " << WSAGetLastError();
#else
        LOG (WARNING) << "Bind failed: " << strerror (errno);
#endif
        CloseSocketHandle (serverSocket);
        serverSocket = INVALID_SOCKET_HANDLE;
        return false;
    }

    // Listen for connections
    if (listen (serverSocket, 5) == SOCKET_ERROR_HANDLE)
    {
#ifdef _WIN32
        LOG (WARNING) << "Listen failed: " << WSAGetLastError();
#else
        LOG (WARNING) << "Listen failed: " << strerror (errno);
#endif
        CloseSocketHandle (serverSocket);
        serverSocket = INVALID_SOCKET_HANDLE;
        return false;
    }

    running = true;
    LOG (DBUG) << "Socket server initialized successfully on port " << port;
    return true;
}

bool SocketServerImpl::acceptNewConnections()
{
    if (!running || serverSocket == INVALID_SOCKET_HANDLE)
    {
        return false;
    }

    struct sockaddr_in clientAddr;
    socklen_t clientAddrLen = sizeof (clientAddr);

    // Non-blocking accept call
    SocketHandle clientSocket = accept (serverSocket, (struct sockaddr*)&clientAddr, &clientAddrLen);

    if (clientSocket != INVALID_SOCKET_HANDLE)
    {
        // Add to our list of client sockets
        clientSockets.push_back (clientSocket);

        // Initialize client data
        clientDataMap[clientSocket] = ClientData{};

        // Send a welcome message while socket is still in blocking mode
        std::string welcome = "Connected to MasterServer server\n";
        send (clientSocket, welcome.c_str(), static_cast<int> (welcome.length()), 0);

        // Now set to non-blocking mode after welcome message is sent
#ifdef _WIN32
        u_long mode = 1;
        ioctlsocket (clientSocket, FIONBIO, &mode);
#else
        int flags = fcntl (clientSocket, F_GETFL, 0);
        if (flags >= 0)
        {
            fcntl (clientSocket, F_SETFL, flags | O_NONBLOCK);
        }
#endif

        LOG (DBUG) << "New client connected, socket: " << clientSocket;
        return true;
    }

    return false;
}

bool SocketServerImpl::checkClientData (SocketHandle clientSocket, bool& disconnected)
{
    disconnected = false;

    if (!running || clientSocket == INVALID_SOCKET_HANDLE)
    {
        return false;
    }

    // Buffer for incoming data
    char buffer[Config::SOCKET_BUFFER_SIZE];

    // Try to receive data (non-blocking)
    int bytesReceived = recv (clientSocket, buffer, sizeof (buffer) - 1, 0);

    if (bytesReceived > 0)
    {
        // Data received - null terminate the buffer
        buffer[bytesReceived] = '\0';

        // Process received data
        std::string data (buffer, bytesReceived);
        LOG (DBUG) << "Received " << bytesReceived << " bytes from client " << clientSocket;

        // Process the data
        processClientData (clientSocket, data);
        return true;
    }
    else if (bytesReceived == 0)
    {
        // Connection closed by client
        LOG (DBUG) << "Client " << clientSocket << " disconnected";

        // Mark as disconnected
        disconnected = true;

        // Clean up client data
        clientDataMap.erase (clientSocket);

        // Close the socket
        CloseSocketHandle (clientSocket);
        return false;
    }
    else
    {
        // Check for EWOULDBLOCK and similar errors which are normal in non-blocking
        int errorCode = 0;
#ifdef _WIN32
        errorCode = WSAGetLastError();
        if (errorCode == WSAEWOULDBLOCK)
            return false;
#else
        errorCode = errno;
        if (errorCode == EAGAIN || errorCode == EWOULDBLOCK)
            return false;
#endif
        // This is an actual error
        LOG (WARNING) << "Socket error on client " << clientSocket << ": " << errorCode;
        disconnected = true;
        clientDataMap.erase (clientSocket);
        CloseSocketHandle (clientSocket);
        return false;
    }
}

void SocketServerImpl::processClientData (SocketHandle clientSocket, const std::string& data)
{
    // Get or create client data
    auto& clientData = clientDataMap[clientSocket];

    // Append new data to any residual data
    std::string fullData = clientData.residualBuffer + data;
    clientData.residualBuffer.clear();

    // Process complete lines
    size_t pos = 0;
    size_t prevPos = 0;

    while ((pos = fullData.find ('\n', prevPos)) != std::string::npos)
    {
        // Extract complete message
        std::string message = fullData.substr (prevPos, pos - prevPos);

        // Add to lock-free queue
        moodyMessages.enqueue (message);
        LOG (DBUG) << "Added to moody queue: " << message;

        // Handle special commands with immediate response
        if (message == "ping")
        {
            std::string response = "pong\n";
            trySendToClient (clientSocket, response);
        }

        // Move to next position
        prevPos = pos + 1;
    }

    // Save any residual data
    if (prevPos < fullData.length())
    {
        clientData.residualBuffer = fullData.substr (prevPos);
    }
}

bool SocketServerImpl::trySendToClient (SocketHandle clientSocket, const std::string& message)
{
    if (clientSocket == INVALID_SOCKET_HANDLE)
        return false;

    auto it = clientDataMap.find (clientSocket);
    if (it == clientDataMap.end())
        return false;

    auto& clientData = it->second;

    // If a send is already in progress, queue the new data
    if (clientData.sendInProgress)
    {
        clientData.sendBuffer += message;
        return true;
    }

    // Try to send the data
    int bytesSent = send (clientSocket, message.c_str(), static_cast<int> (message.length()), 0);

    if (bytesSent == SOCKET_ERROR_HANDLE)
    {
#ifdef _WIN32
        int errorCode = WSAGetLastError();
        if (errorCode == WSAEWOULDBLOCK)
#else
        int errorCode = errno;
        if (errorCode == EAGAIN || errorCode == EWOULDBLOCK)
#endif
        {
            // Would block, save for later
            clientData.sendInProgress = true;
            clientData.sendBuffer = message;
            return true;
        }

        LOG (WARNING) << "Socket send error: " << errorCode;
        return false;
    }
    else if (bytesSent < static_cast<int> (message.length()))
    {
        // Partial send
        clientData.sendInProgress = true;
        clientData.sendBuffer = message.substr (bytesSent);
        return true;
    }

    // Full send succeeded
    return true;
}

void SocketServerImpl::checkPendingSends()
{
    for (auto it = clientDataMap.begin(); it != clientDataMap.end(); /* no increment */)
    {
        SocketHandle socket = it->first;
        ClientData& clientData = it->second;

        if (clientData.sendInProgress && !clientData.sendBuffer.empty())
        {
            int bytesSent = send (socket, clientData.sendBuffer.c_str(),
                                  static_cast<int> (clientData.sendBuffer.length()), 0);

            if (bytesSent == SOCKET_ERROR_HANDLE)
            {
#ifdef _WIN32
                int errorCode = WSAGetLastError();
                if (errorCode == WSAEWOULDBLOCK)
#else
                int errorCode = errno;
                if (errorCode == EAGAIN || errorCode == EWOULDBLOCK)
#endif
                {
                    // Still would block, try later
                    ++it;
                    continue;
                }

                // Actual error
                LOG (WARNING) << "Socket send error: " << errorCode;
                CloseSocketHandle (socket);
                it = clientDataMap.erase (it);
                continue;
            }
            else if (bytesSent < static_cast<int> (clientData.sendBuffer.length()))
            {
                // Partial send, update buffer
                clientData.sendBuffer = clientData.sendBuffer.substr (bytesSent);
                ++it;
            }
            else
            {
                // Complete send
                clientData.sendInProgress = false;
                clientData.sendBuffer.clear();
                ++it;
            }
        }
        else
        {
            // No pending send
            ++it;
        }
    }
}

void SocketServerImpl::checkAllClients()
{
    if (!running || clientSockets.empty())
    {
        return;
    }

    // Using std::erase_if for C++20 or equivalent for older compilers
    // Use a separate vector to collect sockets to remove to avoid O(n²) complexity
    std::vector<SocketHandle> socketsToRemove;

    // Check each client for incoming data
    for (SocketHandle socket : clientSockets)
    {
        bool disconnected = false;
        checkClientData (socket, disconnected);

        if (disconnected)
        {
            socketsToRemove.push_back (socket);
        }
    }

    // Remove disconnected sockets using swap-and-pop for better performance
    for (SocketHandle socket : socketsToRemove)
    {
        auto it = std::find (clientSockets.begin(), clientSockets.end(), socket);
        if (it != clientSockets.end())
        {
            // Swap with last element and pop_back (O(1) removal)
            if (it != clientSockets.end() - 1)
                *it = clientSockets.back();
            clientSockets.pop_back();
        }
    }

    // Check and continue pending sends
    checkPendingSends();
}

void SocketServerImpl::sendResponseToAllClients (const std::string& response)
{
    if (!running || clientSockets.empty())
    {
        return;
    }

    std::string formattedResponse = response;
    if (!formattedResponse.empty() && formattedResponse.back() != '\n')
    {
        formattedResponse += '\n';
    }

    // Send to all connected clients with retry mechanism
    for (SocketHandle socket : clientSockets)
    {
        trySendToClient (socket, formattedResponse);
    }
}

void SocketServerImpl::shutdown()
{
    LOG (DBUG) << "SocketServerImpl::shutdown called";

    running = false;

    // Close client sockets
    for (SocketHandle socket : clientSockets)
    {
        CloseSocketHandle (socket);
    }
    clientSockets.clear();
    clientDataMap.clear();

    // Close server socket
    if (serverSocket != INVALID_SOCKET_HANDLE)
    {
        CloseSocketHandle (serverSocket);
        serverSocket = INVALID_SOCKET_HANDLE;
    }
}