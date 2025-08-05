#pragma once

namespace fs = std::filesystem;

// Centralized service for managing all property systems in the application
// Acts as a container and initializer for different property categories
class PropertyService
{
 public:
    PropertyService() = default;
    ~PropertyService() = default;

    // Initialize all property containers
    void init()
    {
        renderProps = std::make_shared<RenderProperties>();

        // Initialize other property containers as needed
        // ioProps = std::make_shared<IOProperties>();
        // worldProps = std::make_shared<WorldProperties>();
        paintProps = std::make_shared<PaintProperties>();
        physicsProps = std::make_shared<PhysicsProperties>();
        // flexProps = std::make_shared<FlexProperties>();

        resetProperties();
    }

    // Reset all properties to their default values
    void resetProperties()
    {
        initRenderProperties();
        initPathProperties();
        initEnvironmentProperties();
        initPlaybackProperties();
        initCameraProperties();
        initMaterialProperties();
        initPaintProperties(); 
        initPhysicsProperties();
        /*   Initialize other property categories when implementing them
           initIOProperties();
           initWorldProperties();
           initPaintProperties();
           initFlexProperties();*/
    }

    // Initialize rendering-related properties
    void initRenderProperties()
    {
        renderProps->addDefault (RenderKey::ShowPerformanceGraph, true);
        renderProps->addDefault (RenderKey::RenderTime, DEFAULT_RENDER_TIME);
        renderProps->addDefault (RenderKey::RenderPasses, DEFAULT_RENDER_PASSES);
        renderProps->addDefault (RenderKey::RenderMode, DEFAULT_RENDER_MODE);
        renderProps->addDefault (RenderKey::RenderPlaybackType, DEFAULT_RENDER_PLAYBACK_MODE);
        renderProps->addDefault (RenderKey::AutoRender, DEFAULT_AUTO_RENDER);
        renderProps->addDefault (RenderKey::BounceLimit, DEFAULT_BOUNCE_LIMIT);
        renderProps->addDefault (RenderKey::SaveRender, DEFAULT_SAVE_RENDER);
        renderProps->addDefault (RenderKey::SoftwareReleaseMode, DEFAULT_SOFTWARE_RELEASE_MODE);
        renderProps->addDefault (RenderKey::UseEmbeddedPTX, DEFAULT_USE_EMBEDDED_PTX);
        renderProps->addDefault (RenderKey::RenderScale, DEFAULT_RENDER_SCALE);
        renderProps->addDefault (RenderKey::RenderSize, DEFAULT_RENDER_SIZE);
        renderProps->addDefault (RenderKey::RenderDownsampledSize, DEFAULT_RENDER_DOWNSAMPLED_PERCENT);
        renderProps->addDefault (RenderKey::ResizeOnGPU, DEFAULT_RESIZE_ON_GPU);
        renderProps->addDefault (RenderKey::Screengrab, DEFAULT_DO_SCREEN_GRAB);
        renderProps->addDefault (RenderKey::ScreengrabFormat, DEFAULT_SCREENGRAB_FORMAT);
        renderProps->addDefault (RenderKey::UseTimestampedSubfolders, DEFAULT_USE_TIMESTAMPED_SUBFOLDERS);
        renderProps->addDefault (RenderKey::RenderingAnimation, DEFAULT_RENDERING_ANIMATION);
        renderProps->addDefault (RenderKey::CudaTargetArchitectures, DEFAULT_CUDA_GPU_ARCHITECTURES);
        renderProps->addDefault (RenderKey::SelectedGPUIndex, DEFAULT_SELECTED_GPU_INDEX);
        renderProps->addDefault (RenderKey::UseFakeGPUs, DEFAULT_USE_FAKE_GPUS);
        renderProps->addDefault (RenderKey::FakeGPUCount, DEFAULT_FAKE_GPU_COUNT);
        renderProps->addDefault (RenderKey::RenderBuffer, DEFAULT_RENDER_BUFFER);
    }

    // Initialize path-related properties
    void initPathProperties()
    {
        renderProps->addDefault (RenderKey::ResourceFolder, UNSET_PATH);
        renderProps->addDefault (RenderKey::CommonFolder, UNSET_PATH);
        renderProps->addDefault (RenderKey::ContentFolder, UNSET_PATH);
        renderProps->addDefault (RenderKey::RepoFolder, UNSET_PATH);
        renderProps->addDefault (RenderKey::ExternalContentFolder, UNSET_PATH);
        renderProps->addDefault (RenderKey::RootFolder, UNSET_PATH);
        renderProps->addDefault (RenderKey::PtxFolder, UNSET_PATH);
        renderProps->addDefault (RenderKey::FramegrabFolder, UNSET_PATH);
        renderProps->addDefault (RenderKey::TextureFolder, UNSET_PATH);
        renderProps->addDefault (RenderKey::GltfModelFolder, UNSET_PATH);
        renderProps->addDefault (RenderKey::RenderRootFolder, UNSET_PATH);
        renderProps->addDefault (RenderKey::RenderCurrentSubfolder, UNSET_PATH);
        renderProps->addDefault (RenderKey::ScreengrabFolder, UNSET_PATH);
        renderProps->addDefault (RenderKey::PlaybackLoadFolder, UNSET_PATH);
        renderProps->addDefault (RenderKey::HDRImagePath, UNSET_PATH);
    }

    // Initialize environment-related properties
    void initEnvironmentProperties()
    {
        renderProps->addDefault (RenderKey::EnviroRotation, DEFAULT_ENVIRO_ROTATION);
        renderProps->addDefault (RenderKey::EnviroIntensity, DEFAULT_ENVIRO_INTENSITY_PERCENT);
        renderProps->addDefault (RenderKey::RenderEnviro, DEFAULT_RENDER_ENVIRO);
        renderProps->addDefault (RenderKey::BackgroundColor, DEFAULT_BACKGROUND_COLOR);
        renderProps->addDefault (RenderKey::EnviroRotationDirection, DEFAULT_ENVIRO_ROTATE_DIR);
        renderProps->addDefault (RenderKey::EnviroRotationSpeed, DEFAULT_ENVIRO_ROTATE_SPEED);
    }

    // Initialize playback-related properties
    void initPlaybackProperties()
    {
        renderProps->addDefault (RenderKey::LoopPlayback, DEFAULT_LOOP_PLAYBACK);
        renderProps->addDefault (RenderKey::PlaybackDirection, DEFAULT_PLAYBACK_DIR);
        renderProps->addDefault (RenderKey::PlaybackRate, DEFAULT_PLAYBACK_FPS);
        renderProps->addDefault (RenderKey::RenderStartFrame, DEFAULT_RENDER_START_FRAME);
        renderProps->addDefault (RenderKey::RenderEndFrame, DEFAULT_RENDER_END_FRAME);
    }

    // Initialize camera-related properties
    void initCameraProperties()
    {
        renderProps->addDefault (RenderKey::Aperture, DEFAULT_APERTURE);
        renderProps->addDefault (RenderKey::FocalLength, DEFAULT_FOCAL_DIST);
        renderProps->addDefault (RenderKey::CameraPose, Eigen::Affine3f::Identity());
    }

    // Initialize material-related properties
    void initMaterialProperties()
    {
        renderProps->addDefault (RenderKey::MakeGlassScene, DEFAULT_GLASS_SCENE);
        renderProps->addDefault (RenderKey::GlassAbsorption, DEFAULT_GLASS_ABSORPTION);
        renderProps->addDefault (RenderKey::GlassIOR, DEFAULT_GLASS_IOR);
        renderProps->addDefault (RenderKey::GlassType, DEFAULT_GLASS_KIND);
    }

    // Initialize paint-related properties
    void initPaintProperties()
    {
        paintProps->addDefault (PaintKey::toolType, DEFAULT_PAINT_TOOL);
        paintProps->addDefault (PaintKey::toolState, DEFAULT_PAINT_TOOL_STATE);
        paintProps->addDefault (PaintKey::centerOnSurface, DEFAULT_CENTER_ON_SURFACE);
        paintProps->addDefault (PaintKey::axis, DEFAULT_AXIS_TYPE);
        paintProps->addDefault (PaintKey::sense, DEFAULT_SENSE);
        paintProps->addDefault (PaintKey::offsetFromSurface, DEFAULT_OFFSET_FROM_SURFACE);
        paintProps->addDefault (PaintKey::upDirection, DEFAULT_UP_DIRECTION);
        paintProps->addDefault (PaintKey::alignToDrag, DEFAULT_ALIGN_TO_DRAG);
        paintProps->addDefault (PaintKey::dragSense, DEFAULT_DRAG_SENSE);
        paintProps->addDefault (PaintKey::dragSmoothing, DEFAULT_DRAG_SMOOTHING);
        paintProps->addDefault (PaintKey::minScale, DEFAULT_MIN_SCALE);
        paintProps->addDefault (PaintKey::maxScale, DEFAULT_MAX_SCALE);
        paintProps->addDefault (PaintKey::uniformScale, DEFAULT_UNIFORM_SCALE);
        paintProps->addDefault (PaintKey::randomScale, DEFAULT_RANDOM_SCALE);
        paintProps->addDefault (PaintKey::minRotation, DEFAULT_MIN_ROTATION);
        paintProps->addDefault (PaintKey::maxRotation, DEFAULT_MAX_ROTATION);
        paintProps->addDefault (PaintKey::randomRotation, DEFAULT_RANDOM_ROTATION);
        paintProps->addDefault (PaintKey::emitType, DEFAULT_EMIT_TYPE);

        // tweak properties
        paintProps->addDefault (PaintKey::tweakMode, DEFAULT_TWEAK_MODE);

        // paint properties
        paintProps->addDefault (PaintKey::paintMode, DEFAULT_PAINT_MODE);

        // grow properties
        paintProps->addDefault (PaintKey::growCount, DEFAULT_GROW_COUNT);
        paintProps->addDefault (PaintKey::growSpacing, DEFAULT_GROW_SPACING);

        // wall properties
        paintProps->addDefault (PaintKey::wallRows, DEFAULT_WALL_ROWS);
        paintProps->addDefault (PaintKey::wallColumns, DEFAULT_WALL_COLS);
        paintProps->addDefault (PaintKey::wallHorizontalGap, DEFAULT_WALL_HORIZONTAL_GAP);
        paintProps->addDefault (PaintKey::wallVerticalGap, DEFAULT_WALL_VERTICAL_GAP);

        // impulse properties
        paintProps->addDefault (PaintKey::sendImpulseInstances, DEFAULT_SEND_IMPULSE_INSTANCES);
    }


    void initPhysicsProperties()
    {
        // Engine state
        physicsProps->addDefault (PhysicsKeys::PhysicsEngineState, DEFAULT_PHYSICS_ENGINE_STATE);

        // Body properties
        physicsProps->addDefault (PhysicsKeys::BodyType, sabi::DEFAULT_BODY_TYPE);
        physicsProps->addDefault (PhysicsKeys::CollisionShape, sabi::DEFAULT_COLLISION_SHAPE);

        // Mass - using dynamic mass as the default
        physicsProps->addDefault (PhysicsKeys::Mass, sabi::DEFAULT_DYNAMIC_MASS);

        // Material properties
        physicsProps->addDefault (PhysicsKeys::StaticFriction, sabi::DEFAULT_STATIC_FRICTION);
        physicsProps->addDefault (PhysicsKeys::DynamicFriction, sabi::DEFAULT_DYNAMIC_FRICTION);
        physicsProps->addDefault (PhysicsKeys::Bounciness, sabi::DEFAULT_BOUNCINESS);

        // Force and motion
        physicsProps->addDefault (PhysicsKeys::Force, sabi::DEFAULT_FORCE);
        physicsProps->addDefault (PhysicsKeys::ImpulseSpeed, sabi::DEFAULT_IMPULSE_SPEED);
    }

    // Check if a path property is set (non-empty)
    bool isPathSet (RenderKey key) const
    {
        try
        {
            return !renderProps->getValOr<std::string> (key, "").empty();
        }
        catch (...)
        {
            return false;
        }
    }

    // Ensure a path exists by creating directories if needed
    bool ensurePathExists (RenderKey key)
    {
        try
        {
            std::string path = renderProps->getValOr<std::string> (key, "");
            if (path.empty()) return false;

            fs::path dirPath (path);
            if (!fs::exists (dirPath))
            {
                return fs::create_directories (dirPath);
            }
            return true;
        }
        catch (const std::exception& e)
        {
            // LOG(WARNING) << "Failed to create directory: " << e.what();
            return false;
        }
    }

    // Property containers - only RenderProps implemented as requested
    std::shared_ptr<RenderProperties> renderProps = nullptr;

    // Placeholders for other property containers
    // std::shared_ptr<IOProperties> ioProps = nullptr;
    // std::shared_ptr<WorldProperties> worldProps = nullptr;
     std::shared_ptr<PaintProperties> paintProps = nullptr;
     std::shared_ptr<PhysicsProperties> physicsProps = nullptr;
    // std::shared_ptr<FlexProperties> flexProps = nullptr;
};