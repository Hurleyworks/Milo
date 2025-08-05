#pragma once

using mace::InputEvent;

struct init
{
    init (const MessageService& messengers, const PropertyService& properties) :
        messengers (messengers),
        properties (properties)
    {
    }
    QmsID id = QmsID::Init;
    QmsID realID = QmsID::Init;

    MessageService messengers;
    PropertyService properties;
};

struct tick
{
    tick (uint32_t frameNumber = 0) :
        frame (frameNumber)
    {
    }

    QmsID id = QmsID::Tick;
    QmsID realID = QmsID::Tick;

    uint32_t frame = 0;
};

struct rendererReady
{
    rendererReady () 
    {
    }

    QmsID id = QmsID::TopPriority;
    QmsID realID = QmsID::RendererReady;

    uint32_t frame = 0;
};


struct onPriorityInput
{
    onPriorityInput (const InputEvent& input, uint32_t frameNumber = 0) :
        input (input),
        frameNumber (frameNumber)
    {
    }

    // must be responsive to user input!
    QmsID id = QmsID::TopPriority;
    QmsID realID = QmsID::OnPriorityInput;

    InputEvent input;
    uint32_t frameNumber = 0;
};

struct onRenderFrameComplete
{
    onRenderFrameComplete (const std::string& fullPath, uint32_t frameNumber) :
        fullPath (fullPath),
        frameNumber (frameNumber)
    {
    }

    QmsID id = QmsID::TopPriority;
    QmsID realID = QmsID::OnRenderFrameComplete;

    std::string fullPath = UNSET_PATH;
    uint32_t frameNumber = 0;
};

struct onError
{
    onError (const std::string& message, ErrorSeverity severity = ErrorSeverity::Critical) :
        errorMessage (message),
        level (severity)
    {
    }

    QmsID id = QmsID::TopPriority;
    QmsID realID = QmsID::OnError;

    std::string errorMessage;
    ErrorSeverity level;
};
