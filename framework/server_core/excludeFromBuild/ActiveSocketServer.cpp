#include "ActiveSocketServer.h"

ActiveSocketServer::ActiveSocketServer() :
    impl (new SocketServerImpl()),
    stateThread()
{
    // Start the message loop thread
    start();
}

ActiveSocketServer::~ActiveSocketServer()
{
    // Just signal done once and join
    getMessenger().send (qms::clear_queue());
    if (stateThread.joinable())
        stateThread.join();
}
void ActiveSocketServer::start()
{
    stateThread = std::thread (&ActiveSocketServer::messageLoop, this);
}

void ActiveSocketServer::messageLoop()
{
    LOG (DBUG) << "ActiveSocketServer thread is starting up";
    shutdown = false;

    while (!shutdown)
    {
        // Process any incoming messages
        incoming.wait()
            .handle<qms::clear_queue> ([&] (qms::clear_queue const& msg)
                                       { handleClearQueue(); })
            .handle<QMS::initSocketServer> ([&] (QMS::initSocketServer const& msg)
                                            { handleInitSocketServer (msg.port, msg.messengers); })
            .handle<QMS::updateSocketServer> ([&] (QMS::updateSocketServer const& msg)
                                              { handleUpdateSocketServer(); });

        // Brief sleep to prevent tight loops
        std::this_thread::sleep_for (std::chrono::milliseconds (Config::THREAD_SLEEP_MS));
    }

    LOG (DBUG) << "ActiveSocketServer thread is shutting down";
}

void ActiveSocketServer::handleClearQueue()
{
    LOG (DBUG) << "Received clear_queue message";
    shutdown = true;
}

void ActiveSocketServer::handleInitSocketServer (int port, const MessageService& msgService)
{
    LOG (DBUG) << "Initializing socket server on port " << port;

    serverPort = port;
    messengers = msgService;

    // Initialize the socket server
    if (impl)
    {
        bool success = impl->init (serverPort);
        if (success)
        {
            LOG (DBUG) << "Socket server initialized successfully";
        }
        else
        {
            LOG (WARNING) << "Failed to initialize socket server";
            messengers.claude.send (QMS::processMsg ("Error initializing socket server"));
        }
    }
}

void ActiveSocketServer::handleUpdateSocketServer()
{
    // This method is called by the timer from MasterServer
    if (impl)
    {
        // Check for new connections
        impl->acceptNewConnections();

        // Check clients for data
        impl->checkAllClients();
    }
}