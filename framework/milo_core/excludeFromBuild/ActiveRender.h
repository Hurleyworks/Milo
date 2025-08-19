#pragma once

// ActiveRender implements the Active Object pattern for the Milo renderer.
// Runs the renderer in its own thread to decouple rendering from the main thread.
// Communicates via message passing for thread-safe operation.

using mace::InputEvent;
using sabi::CameraHandle;
using sabi::RenderableWeakRef;
using sabi::WeakRenderableList;

class Renderer; // forward decl for impl

class ActiveRender
{

 public:
    ActiveRender();
    ~ActiveRender();

    // messaging
    MsgSender getMessenger() { return incoming; }
    void done() { getMessenger().send (qms::clear_queue()); }

 private:
    std::unique_ptr<Renderer> impl;

    // messaging
    MsgReceiver incoming;

    // message arguments
    uint32_t frameNumber = 0;
    MessageService messengers;
    PropertyService properties;
    InputEvent inputEvent;
    std::filesystem::path path;
    bool updateMotion = false;
    ImageCacheHandlerPtr imageCache = nullptr;
    CameraHandle camera = nullptr;
    RenderableWeakRef weakNode;
    std::string pipelineName;
    std::string engineName;
    int riprRenderMode = 0;

    // state functions
    std::thread stateThread;
    void (ActiveRender::*state)();
    void waitingForMessages();
    void init();
    void initRenderEngine();
    void renderNextFrame();
    void addSkydomeHDR();
    void addWeakNode();
    void setEngine();
    void setRiPRRenderMode();

    // state thread function
    void executeState();
    void start();

    bool shutdown = false;

}; // end class ActiveRender
