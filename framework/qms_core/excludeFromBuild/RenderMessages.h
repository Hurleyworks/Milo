#pragma once

using mace::InputEvent;
using sabi::CameraHandle;
using sabi::RenderableList;
using sabi::RenderableNode;
using sabi::RenderableWeakRef;
using sabi::ViewportCameraPtr;
using sabi::WeakRenderableList;

struct gpuStatsMsg
{
    gpuStatsMsg (GPUMemoryStats stats) :
        stats (stats)
    {
    }

    GPUMemoryStats stats;
    QmsID id = QmsID::TopPriority;
    QmsID realID = QmsID::GPUStatsMsg;
};

struct addWeakNode
{
    addWeakNode (RenderableWeakRef renderable) :
        weakNode (renderable)
    {
    }

    QmsID id = QmsID::AddWeakNode;
    QmsID realID = QmsID::AddWeakNode;

    RenderableWeakRef weakNode;
};

struct removeWeakNode
{
    removeWeakNode (RenderableWeakRef renderable) :
        weakNode (renderable)
    {
    }

    QmsID id = QmsID::RemoveWeakNode;
    QmsID realID = QmsID::RemoveWeakNode;

    RenderableWeakRef weakNode;
};

struct removeWeakNodeByID
{
    removeWeakNodeByID (BodyID bodyID) :
        bodyID (bodyID)
    {
    }

    QmsID id = QmsID::RemoveWeakNodeByID;
    QmsID realID = QmsID::RemoveWeakNodeByID;

    BodyID bodyID;
};

struct removeWeakNodeListByID
{
    removeWeakNodeListByID (const std::vector<BodyID>& bodyIDs) :
        bodyIDs (bodyIDs)
    {
    }

    QmsID id = QmsID::RemoveWeakNodeListByID;
    QmsID realID = QmsID::RemoveWeakNodeListByID;

    std::vector<BodyID> bodyIDs;
};

struct updateNodeMaterial
{
    updateNodeMaterial (RenderableWeakRef renderable) :
        weakNode (renderable)
    {
    }

    QmsID id = QmsID::UpdateNodeMaterial;
    QmsID realID = QmsID::UpdateNodeMaterial;

    RenderableWeakRef weakNode;
};

struct addWeakNodeList
{
    addWeakNodeList (WeakRenderableList&& weakNodes) :
        weakNodes (std::move (weakNodes))
    {
    }

    QmsID id = QmsID::AddWeakNodeList;
    QmsID realID = QmsID::AddWeakNodeList;

    WeakRenderableList weakNodes;
};

struct removedRenderableNodes
{
    removedRenderableNodes (uint32_t count) :
        count(count)
    {
    }

    QmsID id = QmsID::RemovedRenderableNodes;
    QmsID realID = QmsID::RemovedRenderableNodes;

    uint32_t count = 0;
};

struct removeWeakNodeList
{
    removeWeakNodeList (WeakRenderableList&& weakNodes) :
        weakNodes (std::move (weakNodes))
    {
    }

    QmsID id = QmsID::RemoveWeakNodeList;
    QmsID realID = QmsID::RemoveWeakNodeList;

    WeakRenderableList weakNodes;
};

struct initRenderEngine
{
    initRenderEngine (const CameraHandle& camera, ImageCacheHandlerPtr imageCache) :
        camera (camera),
        imageCache (imageCache)
    {
    }
    QmsID id = QmsID::InitRenderEngine;
    QmsID realID = QmsID::InitRenderEngine;

    CameraHandle camera = nullptr;
    ImageCacheHandlerPtr imageCache = nullptr;
};

struct renderNextFrame
{
    renderNextFrame (const InputEvent& inputEvent, bool updateMotion, uint32_t frameNumber = 0) :
        input (inputEvent),
        updateMotion (updateMotion),
        frameNumber (frameNumber)
    {
    }
    QmsID id = QmsID::RenderNextFrame;
    QmsID realID = QmsID::RenderNextFrame;

    InputEvent input;
    bool updateMotion = false;
    uint32_t frameNumber = 0;
};

struct renderedFrameComplete
{
    renderedFrameComplete(float renderTime) :
        renderTime(renderTime)
    {

    }

     QmsID id = QmsID::TopPriority;
    QmsID realID = QmsID::RenderedFrameComplete;

    float renderTime = 0.0f;
};

struct renderNextAnimationFrame
{
    renderNextAnimationFrame (const InputEvent& inputEvent, bool updateMotion, uint32_t frameNumber = 0) :
        input (inputEvent),
        updateMotion (updateMotion),
        frameNumber (frameNumber)
    {
    }
    // Make sure animation frames aren't eaten by the dispatcher
    // unlike preview mode, they all need to pass on to the renderer
    QmsID id = QmsID::TopPriority;
    QmsID realID = QmsID::RenderNextFrame;

    InputEvent input;
    bool updateMotion = false;
    uint32_t frameNumber = 0;
};

struct addSkydomeHDR
{
    addSkydomeHDR (const std::filesystem::path& hdrPath) :
        path (hdrPath)
    {
    }
    QmsID id = QmsID::AddSkydomeHDR;
    QmsID realID = QmsID::AddSkydomeHDR;

    std::filesystem::path path;
};

struct resetMotion
{
    resetMotion()
    {
    }
    QmsID id = QmsID::TopPriority;
    QmsID realID = QmsID::RenderNextFrame;
};

struct setAllModelsVisibity
{
    setAllModelsVisibity (uint32_t mask) :
        visibilityMask (mask)
    {
    }
    QmsID id = QmsID::TopPriority;
    QmsID realID = QmsID::SetAllModelsVisibility;

    uint32_t visibilityMask = 0;
};

struct updateViewportCameras
{
    updateViewportCameras (const std::vector<ViewportCameraPtr>& cameras) :
        viewportCameras (cameras)
    {
    }
    QmsID id = QmsID::TopPriority;
    QmsID realID = QmsID::UpdateViewportCameras;

    std::vector<ViewportCameraPtr> viewportCameras;
};

struct renderableNodeProcessed
{
    renderableNodeProcessed (BodyID nodeID) :
        nodeID (nodeID)
    {
    }
    QmsID id = QmsID::TopPriority; // High priority to ensure quick processing
    QmsID realID = QmsID::RenderableNodeProcessed;

    BodyID nodeID;
};

// In RenderMessages.h
struct listAvailableGPUs
{
    listAvailableGPUs() {}

    QmsID id = QmsID::ListAvailableGPUs;
    QmsID realID = QmsID::ListAvailableGPUs;
};

struct gpuEnumeration
{
    gpuEnumeration (const std::vector<std::string>& names,
                    const std::vector<size_t>& memory,
                    int selected) :
        gpuNames (names),
        gpuMemorySizes (memory),
        selectedIndex (selected)
    {
    }

    QmsID id = QmsID::TopPriority;
    QmsID realID = QmsID::ListAvailableGPUs;

    std::vector<std::string> gpuNames;
    std::vector<size_t> gpuMemorySizes;
    int selectedIndex;
};

struct selectGPU
{
    selectGPU (int index) :
        gpuIndex (index)
    {
    }

    QmsID id = QmsID::SelectGPU;
    QmsID realID = QmsID::SelectGPU;

    int gpuIndex;
};

struct setPipeline
{
    setPipeline (const std::string& pipelineName) :
        pipelineName (pipelineName)
    {
    }

    QmsID id = QmsID::SetPipeline;
    QmsID realID = QmsID::SetPipeline;

    std::string pipelineName;
};

struct setEngine
{
    setEngine (const std::string& engineName) :
        engineName (engineName)
    {
    }

    QmsID id = QmsID::SetEngine;
    QmsID realID = QmsID::SetEngine;

    std::string engineName;
};

struct enablePipelineSystem
{
    enablePipelineSystem (bool enable) :
        enable (enable)
    {
    }

    QmsID id = QmsID::EnablePipelineSystem;
    QmsID realID = QmsID::EnablePipelineSystem;

    bool enable;
};

struct setRiPRRenderMode
{
    setRiPRRenderMode (int mode) :
        mode (mode)
    {
    }

    QmsID id = QmsID::SetRiPRRenderMode;
    QmsID realID = QmsID::SetRiPRRenderMode;

    int mode;
};