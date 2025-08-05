// Updated ClaudeMessages.h
#pragma once

struct initSocketServer
{
    QmsID id = QmsID::InitSocketServer;
    QmsID realID = QmsID::InitSocketServer;
    int port;
    MessageService messengers;

    initSocketServer (int p, MessageService ms) :
        port (p), messengers (ms) {}
};

struct clientConnected
{
    QmsID id = QmsID::ClientConnected;
    QmsID realID = QmsID::ClientConnected;
    SocketHandle clientSocket;

    clientConnected (SocketHandle socket) :
        clientSocket (socket) {}
};

struct processMsg
{
    QmsID id = QmsID::ProcessMsg;
    QmsID realID = QmsID::ProcessMsg;
    std::string message;

    processMsg (const std::string& msg) :
        message (msg) {}
};

struct sendMsg
{
    QmsID id = QmsID::SendMsg;
    QmsID realID = QmsID::SendMsg;
    SocketHandle clientSocket;
    std::string message;

    sendMsg (SocketHandle socket, const std::string& msg) :
        clientSocket (socket), message (msg) {}
};

struct broadcastMsg
{
    QmsID id = QmsID::BroadcastMsg;
    QmsID realID = QmsID::BroadcastMsg;
    std::string message;

    broadcastMsg (const std::string& msg) :
        message (msg) {}
};

struct updateSocketServer
{
    QmsID id = QmsID::TopPriority;
    QmsID realID = QmsID::UpdateSocketServer;

    updateSocketServer() {}
};

struct executeCommand
{
    QmsID id = QmsID::ExecuteCommand;
    QmsID realID = QmsID::ExecuteCommand;

    executeCommand (const std::string& cmd) :
        command (cmd) {}
    std::string command;
};

struct executeCommandRequest
{
    QmsID id = QmsID::ExecuteCommandRequest;
    QmsID realID = QmsID::ExecuteCommandRequest;

    executeCommandRequest (const std::string& cmd) :
        command (cmd) {}
    std::string command;
};

struct commandResult
{
    QmsID id = QmsID::CommandResult;
    QmsID realID = QmsID::CommandResult;

    commandResult (bool s, const std::string& cmd, const std::string& resp) :
        success (s), command (cmd), response (resp) {}
    bool success;
    std::string command;
    std::string response;
};

struct initCommandProcessor
{
    QmsID id = QmsID::InitCommandProcessor;
    QmsID realID = QmsID::InitCommandProcessor;

    initCommandProcessor (MessageService m) :
        messengers (m) {}
    MessageService messengers;
};