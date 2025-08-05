
#pragma once

#include "SocketServerImpl.h"

class ActiveSocketServer
{
 public:
    ActiveSocketServer();
    ~ActiveSocketServer();

    // Get the messenger for this active object - now returning by reference
    MsgSender getMessenger() { return incoming; }

    // Signal that we're done, used to shut down the active object
    void done() { getMessenger().send (qms::clear_queue()); }

    // Access to the underlying implementation
    SocketServerImpl* getServer() { return impl.get(); }

 private:
    // Socket server implementation
    std::unique_ptr<SocketServerImpl> impl;

    // Messaging
    MsgReceiver incoming;
    MessageService messengers;

    // Thread management
    std::thread stateThread;
    std::atomic<bool> shutdown{false}; // Now atomic

    // Server data
    int serverPort = Config::DEFAULT_SOCKET_PORT;

    // Thread function that processes messages
    void messageLoop();
    void start();

    // Message handling functions
    void handleUpdateSocketServer();
    void handleClearQueue();
    void handleInitSocketServer (int port, const MessageService& msgService);
};