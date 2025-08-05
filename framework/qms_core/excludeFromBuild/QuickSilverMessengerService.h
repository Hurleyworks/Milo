#pragma once

// quicksilver messenger service
#include "QmsData.h"
#include "QmsQueue.h"
#include "QmsSender.h"
#include "QmsDispatcherT.h"
#include "QmsDispatcher.h"
#include "QmsReceiver.h"

using MsgSender = qms::sender;
using MsgReceiver = qms::receiver;

static const char* MessageDestinationTable[] =
    {
        "Dreamer",
        "Renderer",
        "World",
        "Geometry",
        "Imaging",
        "Physics",
        "App",
        "Invalid"};

struct MessageDestination
{
    enum EMessageDestination
    {
        Dreamer,
        Renderer,
        World,
        Geometry,
        Imaging,
        Physics,
        App,
        Count,
        Invalid = Count
    };

    union
    {
        EMessageDestination name;
        unsigned int value;
    };

    MessageDestination (EMessageDestination name) :
        name (name) {}
    MessageDestination (unsigned int value) :
        value (value) {}
    MessageDestination() :
        value (Invalid) {}
    operator EMessageDestination() const { return name; }
    const char* toString() const { return MessageDestinationTable[value]; }
};

struct MessageService
{
    MsgSender dreamer;
    MsgSender render;
    MsgSender world;
    MsgSender geometry;
    MsgSender imaging;
    MsgSender physics;
    //MsgSender app;
    MsgSender claude;
};

namespace qms
{
    namespace activeMsg
    {
        #include "DreamerMessages.h"
        #include "GeometryMessages.h"
        #include "ImagingMessages.h"
        #include "RenderMessages.h"
        #include "WorldMessages.h"
        #include "PhysicsMessages.h"
        #include "AppMessages.h"
        #include "ClaudeMessages.h"

    } // namespace activeMsg

} // namespace qms

#define QMS qms::activeMsg