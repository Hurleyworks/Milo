#pragma once

#include "Standard.h"

// Forward declarations
class View;

using sabi::CameraBody;
using sabi::CameraHandle;

// Signal type for HDR image changes
using HdrImageSignal = Nano::Signal<void (const std::filesystem::path&)>;
using LoadGLTFSignal = Nano::Signal<void (const std::filesystem::path&)>;
using LoadGltfFolderSignal = Nano::Signal<void (const std::filesystem::path&)>;

class CommandProcessor
{
 public:
    // Signals
    HdrImageSignal hdrImageChangeEmitter;
    LoadGLTFSignal loadGltfEmitter;
    LoadGltfFolderSignal loadGltfFolderEmitter;

 public:
    CommandProcessor (View* gui, const PropertyService& properties);
    ~CommandProcessor() = default;

    // Main command processing entry point
    std::string processCommand (const std::string& cmd);

 private:
    View* gui;
    PropertyService properties;

    // Command handlers
    std::string processPingCommand();
    std::string processCameraInfoCommand();
    std::string processCameraLookAtCommand (const std::string& cmd);
    std::string processCameraZoomCommand (const std::string& cmd);
    std::string processCameraPanCommand (const std::string& cmd);
    std::string processBackgroundColorCommand (const std::string& cmd);
    std::string processCubeRotationCommand (const std::string& cmd);
    std::string processSetRenderPassesCommand (const std::string& cmd);
    std::string processSetPreviewScaleCommand (const std::string& cmd);
    std::string processSetHDRImageCommand (const std::string& cmd);
    std::string processSetContentDirectoryCommand (const std::string& cmd);
    std::string processGetRenderSettingsCommand();
    std::string processLoadGLTFCommand (const std::string& cmd);
    std::string processLoadGltfFolderCommand (const std::string& cmd);
    std::string processSetPipelineCommand (const std::string& cmd);
    std::string processGetAvailablePipelinesCommand();

    // Helper methods
    bool validateColorValues (int r, int g, int b, int a);
    CameraHandle getCamera();
};