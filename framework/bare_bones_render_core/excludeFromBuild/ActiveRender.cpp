#include "ActiveRender.h"
#include "Renderer.h"
#include <g3log/g3log.hpp>

using namespace qms;

// Constructor
ActiveRender::ActiveRender() :
    impl(new Renderer()),
    stateThread()
{
    start();
}

// Destructor
ActiveRender::~ActiveRender()
{
    done();
    if (stateThread.joinable())
        stateThread.join();
}

// Start the state thread
void ActiveRender::start()
{
    stateThread = std::thread(&ActiveRender::executeState, this);
}

// Main state machine execution
void ActiveRender::executeState()
{
    LOG(DBUG) << "ActiveRender thread is starting up";
    
    state = &ActiveRender::waitingForMessages;
    
    while (!shutdown)
    {
        (this->*state)();
    }
    
    if (impl)
    {
        impl->finalize();
    }
    
    LOG(DBUG) << "ActiveRender thread is shutting down";
}

// State: Waiting for messages
void ActiveRender::waitingForMessages()
{
    incoming.wait()
        .handle<qms::clear_queue>([&](qms::clear_queue const& msg)
            { 
                shutdown = true; 
            })

        .handle<QMS::init>([&](QMS::init const& msg)
            {
                messengers = msg.messengers;
                properties = msg.properties;
                state = &ActiveRender::init; 
            })

        .handle<QMS::initRenderEngine>([&](QMS::initRenderEngine const& msg)
            {
                imageCache = msg.imageCache;
                camera = msg.camera;
                state = &ActiveRender::initRenderEngine; 
            })
            
        .handle<QMS::addWeakNode>([&](QMS::addWeakNode const& msg)
            {
                weakNode = msg.weakNode;
                state = &ActiveRender::addWeakNode; 
            })

        .handle<QMS::renderNextFrame>([&](QMS::renderNextFrame const& msg)
            { 
                updateMotion = msg.updateMotion;
                inputEvent = msg.input;
                frameNumber = msg.frameNumber;
                state = &ActiveRender::renderNextFrame; 
            })

        .handle<QMS::addSkydomeHDR>([&](QMS::addSkydomeHDR const& msg)
            {
                path = msg.path;
                state = &ActiveRender::addSkydomeHDR; 
            })

        .handle<QMS::setEngine>([&](QMS::setEngine const& msg)
            {
                engineName = msg.engineName;
                state = &ActiveRender::setEngine; 
            });
}

// State: init
void ActiveRender::init()
{
    try
    {
        impl->init(messengers, properties);
    }
    catch (std::exception& e)
    {
        done();
        LOG(WARNING) << e.what();
        messengers.dreamer.send(QMS::onError(e.what() + std::string(" ActiveRender thread is shutting down")));
    }
    catch (...)
    {
        done();
        LOG(WARNING) << "Caught unknown exception!";
        messengers.dreamer.send(QMS::onError("Caught unknown exception!"));
    }
    state = &ActiveRender::waitingForMessages;
}

// State: initRenderEngine
void ActiveRender::initRenderEngine()
{
    try
    {
        if (imageCache && camera)
            impl->initializeEngine(camera, imageCache);
    }
    catch (std::exception& e)
    {
        done();
        LOG(WARNING) << e.what();
        messengers.dreamer.send(QMS::onError(e.what() + std::string(" ActiveRender thread is shutting down")));
    }
    catch (...)
    {
        done();
        LOG(WARNING) << "Caught unknown exception!";
        messengers.dreamer.send(QMS::onError("Caught unknown exception!"));
    }
    state = &ActiveRender::waitingForMessages;
}

// State: renderNextFrame
void ActiveRender::renderNextFrame()
{
    try
    {
        impl->render(inputEvent, updateMotion, frameNumber);

        // reset input event
        inputEvent = InputEvent{};
    }
    catch (std::exception& e)
    {
        done();
        LOG(WARNING) << e.what();
        messengers.dreamer.send(QMS::onError(e.what() + std::string(" ActiveRender thread is shutting down")));
    }
    catch (...)
    {
        done();
        LOG(WARNING) << "Caught unknown exception!";
        messengers.dreamer.send(QMS::onError("Caught unknown exception!"));
    }
    state = &ActiveRender::waitingForMessages;
}

// State: addSkydomeHDR
void ActiveRender::addSkydomeHDR()
{
    try
    {
        // We are allowing invalid paths now
        // and will create default blue hdr image
        // if the path is invalid
        impl->addSkyDomeHDR(path);
    }
    catch (std::exception& e)
    {
        done();
        LOG(WARNING) << e.what();
        messengers.dreamer.send(QMS::onError(e.what() + std::string(" ActiveRender thread is shutting down")));
    }
    catch (...)
    {
        done();
        LOG(WARNING) << "Caught unknown exception!";
        messengers.dreamer.send(QMS::onError("Caught unknown exception!"));
    }
    state = &ActiveRender::waitingForMessages;
}

// State: addWeakNode
void ActiveRender::addWeakNode()
{
    try
    {
        if (!weakNode.expired())
            impl->addRenderableNode(weakNode);
    }
    catch (std::exception& e)
    {
        done();
        LOG(WARNING) << e.what();
        messengers.dreamer.send(QMS::onError(e.what() + std::string(" ActiveRender thread is shutting down")));
    }
    catch (...)
    {
        done();
        LOG(WARNING) << "Caught unknown exception!";
        messengers.dreamer.send(QMS::onError("Caught unknown exception!"));
    }
    state = &ActiveRender::waitingForMessages;
}

// State: setEngine
void ActiveRender::setEngine()
{
    try
    {
        // Stub implementation for now - in full implementation would call impl->setEngine(engineName)
        LOG(INFO) << "ActiveRender::setEngine - stub for engine: " << engineName;
    }
    catch (std::exception& e)
    {
        done();
        LOG(WARNING) << e.what();
        messengers.dreamer.send(QMS::onError(e.what() + std::string(" ActiveRender thread is shutting down")));
    }
    catch (...)
    {
        done();
        LOG(WARNING) << "Caught unknown exception!";
        messengers.dreamer.send(QMS::onError("Caught unknown exception!"));
    }
    state = &ActiveRender::waitingForMessages;
}