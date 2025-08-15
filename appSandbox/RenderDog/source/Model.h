
#pragma once

#include "ActiveFramework.h"

using sabi::CameraHandle;
using sabi::MeshOptions;
using sabi::PhysicsEngineState;
using sabi::RenderableNode;
using sabi::RenderableList;
using sabi::LoadStrategyPtr;
 
using OnPhyicsEngineChangeSignal = Nano::Signal<void (PhysicsEngineState)>;
using OnHDRLoadedSignal = Nano::Signal<void (const std::filesystem::path&)>;

class Model : public Observer
{
 public:
    OnPhyicsEngineChangeSignal physicsStateEmitter;
    OnHDRLoadedSignal hdrLoadedEmitter;

 public:
    Model();
    ~Model();

    void initialize (CameraHandle camera, const PropertyService& properties);

    // Adds a node to the backend renderere
    void addNodeToRenderer (RenderableNode node);

    void createInstanceStack (uint32_t instanceCount)
    {
       // framework.world.getMessenger().send (QMS::createInstanceStack (instanceCount));
    }

    void loadGLTF (const std::filesystem::path& gltfPath)
    {
        // Temporarily pause animation while adding new geometry
        bool wasAnimating = animationEnabled;
        if (wasAnimating)
        {
            animationEnabled = false;
            LOG(INFO) << "Pausing animation while loading new model";
        }
        
        GLTFImporter gltf;
        // std::vector<Animation> animations;
        auto [cgModel, animations] = gltf.importModel (gltfPath.generic_string());
        if (!cgModel)
        {
            LOG (WARNING) << "Load failed " << gltfPath.string();
            return;
        }

        RenderableNode node = sabi::WorldItem::create();
        node->setClientID (node->getID());
        node->setModel (cgModel);
        node->getState().state |= sabi::PRenderableState::Visible;
        
        // Set the model path in the description so texture loading can find the content folder
        sabi::RenderableDesc desc = node->description();
        desc.modelPath = gltfPath;
        node->setDescription(desc);
        
        std::string modelName = getFileNameWithoutExtension (gltfPath);

        for (auto& s : cgModel->S)
        {
            s.vertexCount = cgModel->vertexCount();
        }
        node->setName (modelName);
        node->getSpaceTime().worldTransform.translation() = Eigen::Vector3f (0.0f, 1.0f, 0.0f);

        // Check if model name starts with "static" - if so, skip processCgModel
        //if (modelName.substr(0, 6) != "static")
        //{
        //    // Use the full mesh options including RestOnGround and LoadStrategy
        //    sabi::MeshOps::processCgModel (node, meshOptions, loadStrategy);
        //}
        //else
        //{
        //    LOG(INFO) << "Skipping processCgModel for static model: " << modelName;
        //}

        // store it so it doesn't self-destruct
        nodes.push_back (node);

        addNodeToRenderer (node);
        
        // Resume animation if it was running before
        if (wasAnimating)
        {
            animationEnabled = true;
            LOG(INFO) << "Resuming animation after loading model";
        }
    }

    void onDrop (const std::vector<std::string>& filenames);
    void onUpdate (const InputEvent& inputEvent)
    {
        bool updateMotion = engineState == PhysicsEngineState (PhysicsEngineState::Start) || engineState == PhysicsEngineState (PhysicsEngineState::Reset);
        
        // Apply rotation animation if enabled
        if (animationEnabled && !nodes.empty())
        {
            
            // Rotation speed: 90 degrees per second, assuming 60 FPS
            const float rotationSpeed = (M_PI * 0.5f) / 60.0f;
            rotationAngle += rotationSpeed;
            
            // Keep rotation in [0, 2*PI] range
            if (rotationAngle > 2.0f * M_PI)
                rotationAngle -= 2.0f * M_PI;
            
            // Calculate sin and cos once
            float cosTheta = cosf(rotationAngle);
            float sinTheta = sinf(rotationAngle);
            
            // Update each node's SpaceTime
            for (auto& node : nodes)
            {
                if (node)
                {
                    sabi::SpaceTime& spacetime = node->getSpaceTime();
                    
                    // Create Y-axis rotation matrix for current angle
                    Eigen::Matrix3f yRotation;
                    yRotation << cosTheta, 0, sinTheta,
                                0, 1, 0,
                                -sinTheta, 0, cosTheta;
                    
                    // Get the current position
                    Eigen::Vector3f position = spacetime.worldTransform.translation();
                    
                    // Create new transform with rotation and original position
                    Eigen::Affine3f newTransform = Eigen::Affine3f::Identity();
                    newTransform.linear() = yRotation;
                    newTransform.translation() = position;
                    
                    spacetime.worldTransform = newTransform;
                }
            }
            
            // We have motion when animating
            updateMotion = true;
        }

        // render next frame async
        framework.render.getMessenger().send (QMS::renderNextFrame (inputEvent, updateMotion, frameNumber++));

        //// update physics async
        //if (engineState != PhysicsEngineState (PhysicsEngineState::Pause))
        //    framework.newton.getMessenger().send (QMS::updatePhysics (engineState));

        // after a Reset, change the engine state back to Pause
        // and update the renderer
        if (engineState == PhysicsEngineState (PhysicsEngineState::Reset))
        {
            engineState = PhysicsEngineState (PhysicsEngineState::Pause);

            // the View's version of PhysicsEngineState is still set to Reset
            // so we need to set it to Pause
            physicsStateEmitter.fire (engineState);
        }
    }

    void onPriorityInput (const InputEvent& inputEvent)
    {
        // Handle mouse press and release, which are vital to painting and picking,
        // with TopPriority so they are not eaten by the Render message dispatcher.
        if (inputEvent.getType() == InputEvent::Type::Press || inputEvent.getType() == InputEvent::Type::Release)
        {
            // LOG (DBUG) << (inputEvent.getType() == InputEvent::Type::Release);
            //  FIXME
            uint32_t renderFrameNumber = 0;
            framework.render.getMessenger().send (QMS::onPriorityInput (inputEvent, renderFrameNumber));
        }
    }

    void setAllModelsVisibility (uint32_t mask)
    {
        LOG (DBUG) << " sending visibility mask from Model " << mask;
        framework.render.getMessenger().send (QMS::setAllModelsVisibity (mask));
    }

    void deselectAll()
    {
        framework.render.getMessenger().send (QMS::deselectAll());
    }

    void selectAll()
    {
        framework.render.getMessenger().send (QMS::selectAll());
    }

    void addSkyDomeImage (const std::filesystem::path& hdrPath)
    {
        framework.render.getMessenger().send (QMS::addSkydomeHDR (hdrPath));
        hdrLoadedEmitter.fire(hdrPath);
    }

    void setPipeline (const std::string& pipelineName)
    {
        framework.render.getMessenger().send (QMS::setPipeline (pipelineName));
    }

    void enablePipelineSystem (bool enable)
    {
        framework.render.getMessenger().send (QMS::enablePipelineSystem (enable));
    }

    void setEngine (const std::string& engineName)
    {
        framework.render.getMessenger().send (QMS::setEngine (engineName));
    }
    
    void setRiPRRenderMode (int mode)
    {
        framework.render.getMessenger().send (QMS::setRiPRRenderMode (mode));
    }

    // load the icon images for the gui
    void loadIcons (const std::string& iconFolder)
    {
        std::vector<std::string> filenames;
        filenames.push_back (iconFolder);
        onDrop (filenames);
    }

    void setPhysicsEngineSate (PhysicsEngineState state) { engineState = state; }
    Dreamer* getDreamer() { return framework.dreamer.getDreamer(); }

    void loadHDRIfromIcon (const std::filesystem::path& iconPath);
    void loadCgModelfromIcon (const std::filesystem::path& iconPath);
    
    void setAnimationEnabled(bool enabled) { animationEnabled = enabled; /* Don't reset angle */ }
    bool isAnimationEnabled() const { return animationEnabled; }

 private:
    ActiveFramework framework;
    ImageCacheHandlerPtr imageCache = nullptr;
    PhysicsEngineState engineState = PhysicsEngineState::Pause;
    PropertyService properties;
    CameraHandle camera = nullptr;

    PathList modelPaths;
    PathList hdrIconPaths;
    PathList modelIconPaths;
    uint32_t frameNumber = 0;

    RenderableList nodes;

    RenderableNode warmLight = nullptr;
    RenderableNode groundPlane = nullptr;

    MeshOptions meshOptions = MeshOptions::CenterVertices | MeshOptions::NormalizeSize | MeshOptions::RestOnGround | MeshOptions::LoadStrategy;
    LoadStrategyPtr loadStrategy = nullptr;
    
    bool animationEnabled = false;
    float rotationAngle = 0.0f;

    void processPath (const std::filesystem::path& p);
};
