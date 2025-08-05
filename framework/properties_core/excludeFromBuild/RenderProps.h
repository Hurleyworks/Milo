#pragma once

enum class ImageFileFormat
{
    EXR, // High quality, HDR support
    PNG, // Lossless compression
    JPG, // High quality JPEG
    Count,
    Invalid = Count
};

// Modern enum classes for render-related types
enum class RenderMode
{
    Preview,
    Render,
    Count,
    Invalid = Count
};

enum class RenderBuffer
{
    Beauty,
    Albedo,
    Normal,
    Count,
    Invalid = Count
};

enum class RenderPlaybackMode
{
    Normal,
    Progressive,
    Count,
    Invalid = Count
};

enum class EnviroRotDir
{
    CW,
    Stop,
    CCW,
    Count,
    Invalid = Count
};

enum class GlassKind
{
    ThinWalled,
    Solid,
    Count,
    Invalid = Count
};

enum class PlaybackDirection
{
    Forward,
    Pause,
    Reverse,
    Count,
    Invalid = Count
};

enum class PreviewScaleFactor
{
    x1,
    x2,
    x3,
    x4,
    Count,
    Invalid = Count
};

enum class RenderDownsizeFactor
{
    One,
    Half,
    Third,
    Quarter,
    Count,
    Invalid = Count
};

// Comprehensive enum class for render property keys
enum class RenderKey
{
    ResourceFolder,
    CommonFolder,
    ContentFolder,
    RepoFolder,
    RenderRootFolder,
    RenderCurrentSubfolder,
    UseTimestampedSubfolders,
    RenderStartFrame,
    RenderEndFrame,
    RenderScale,
    RenderPlaybackType,
    RenderingAnimation,
    ResizeOnGPU,
    PlaybackLoadFolder,
    ExternalContentFolder,
    RootFolder,
    RenderTime,
    RenderState,
    MotionBlur,
    RenderMode,
    RenderBuffer,
    Renderer,
    RenderOutput,
    RenderSize,
    RenderDownsampledSize,
    RenderPasses,
    Screengrab,
    ScreengrabFolder,
    ScreengrabFormat, // Format selection (EXR, PNG, JPG)
    LoopPlayback,
    PlaybackDirection,
    BounceLimit,
    PlaybackRate,
    ShowPerformanceGraph,
    PtxFolder,
    UseEmbeddedPTX,
    FramegrabFolder,
    BackgroundColor,
    FocalLength,
    Aperture,
    ShowParticles,
    ShowBodies,
    HDRImagePath,
    EnviroIntensity,
    EnviroRotation,
    EnviroRotationDirection,
    EnviroRotationSpeed,
    RenderEnviro,
    CameraEyePoint,
    CameraTarget,
    CameraPose,
    VisualizeRays,
    AutoRender,
    TextureFolder,
    SaveRender,
    SoftwareReleaseMode,
    GltfModelFolder,
    MakeGlassScene,
    GlassAbsorption,
    GlassIOR,
    GlassType,
    MaxRadiance,  // Maximum radiance value for firefly clamping
    AreaLightPower,  // Area light power coefficient
    EnableAreaLights,  // Enable/disable area light sampling

    // GPU stats
    GPUusedMemory,
    GPUfreeMemory,
    GPUtotalMemory,
    GPUusagePercent,

    CudaTargetArchitectures,
    SelectedGPUIndex,
    GPUCount,
    GPUName,
    UseFakeGPUs,  // Add this new property for testing
    FakeGPUCount, // Add this to control how many fake GPUs to create

    Count,
    Invalid = Count
};

// Type definitions for render property collection
using RenderProperties = AnyValue<RenderKey>;
using RenderPropsRef = std::shared_ptr<RenderProperties>;

// Constants for default values - now using std::string for path values
// Render settings
const RenderMode DEFAULT_RENDER_MODE = RenderMode::Preview;
const RenderPlaybackMode DEFAULT_RENDER_PLAYBACK_MODE = RenderPlaybackMode::Normal;
const PreviewScaleFactor DEFAULT_RENDER_SCALE = PreviewScaleFactor::x1;
constexpr uint32_t DEFAULT_RENDER_PASSES = 1;
constexpr uint32_t DEFAULT_BOUNCE_LIMIT = 30;
constexpr bool DEFAULT_RESIZE_ON_GPU = true;
constexpr bool DEFAULT_AUTO_RENDER = false;
constexpr bool DEFAULT_SAVE_RENDER = false;
constexpr bool DEFAULT_VISUALIZE_RAYS = false;
constexpr bool DEFAULT_SOFTWARE_RELEASE_MODE = false;
constexpr bool DEFAULT_USE_EMBEDDED_PTX = false;
constexpr bool DEFAULT_RENDERING_ANIMATION = false;
constexpr int DEFAULT_SELECTED_GPU_INDEX = 0;
constexpr bool DEFAULT_USE_FAKE_GPUS = false;
constexpr int DEFAULT_FAKE_GPU_COUNT = 2;
const RenderBuffer DEFAULT_RENDER_BUFFER = RenderBuffer::Beauty;

// Resolution and output settings
const Eigen::Vector2i DEFAULT_RENDER_SIZE = Eigen::Vector2i (1280, 720);
constexpr double DEFAULT_RENDER_DOWNSAMPLED_PERCENT = 1.0;
const std::string DEFAULT_HDRI_PATH = UNSET_PATH;
const std::string DEFAULT_RENDER_ROOT_FOLDER = UNSET_PATH;
const std::string DEFAULT_REPO_FOLDER = UNSET_PATH;
const std::string DEFAULT_RENDER_CURRENT_SUBFOLDER = UNSET_PATH;
const std::string DEFAULT_SCREEN_GRAB_FOLDER = UNSET_PATH;
const ImageFileFormat DEFAULT_SCREENGRAB_FORMAT = ImageFileFormat::EXR;
constexpr bool DEFAULT_DO_SCREEN_GRAB = false;
constexpr bool DEFAULT_USE_TIMESTAMPED_SUBFOLDERS = true;

// Camera settings
constexpr double DEFAULT_APERTURE = 0.0;
constexpr double DEFAULT_FOCAL_DIST= 5.0;
constexpr bool DEFAULT_MOTION_BLUR = false;

// Environment and background settings
constexpr bool DEFAULT_RENDER_ENVIRO = true;
constexpr double DEFAULT_ENVIRO_INTENSITY_PERCENT = 1.0;
constexpr double DEFAULT_ENVIRO_ROTATION = 0.0;
constexpr double DEFAULT_ENVIRO_ROTATE_SPEED = 0.0;
const EnviroRotDir DEFAULT_ENVIRO_ROTATE_DIR = EnviroRotDir::CW;
const Eigen::Vector3d DEFAULT_BACKGROUND_COLOR = Eigen::Vector3d::Zero();

// Glass material settings
constexpr bool DEFAULT_GLASS_SCENE = false;
constexpr double DEFAULT_GLASS_ABSORPTION = 1.0;
constexpr double DEFAULT_GLASS_IOR = 1.52;
const GlassKind DEFAULT_GLASS_KIND = GlassKind::ThinWalled;

// Firefly reduction settings
constexpr float DEFAULT_MAX_RADIANCE = 10.0f;  // Maximum radiance value to clamp fireflies

// Area light settings
constexpr float DEFAULT_AREA_LIGHT_POWER = 10.0f;  // Area light power coefficient
constexpr bool DEFAULT_ENABLE_AREA_LIGHTS = true;  // Enable area lights by default

// Playback settings
constexpr uint32_t DEFAULT_PLAYBACK_FPS = 30;
constexpr bool DEFAULT_LOOP_PLAYBACK = true;
const PlaybackDirection DEFAULT_PLAYBACK_DIR = PlaybackDirection::Forward;
constexpr uint32_t DEFAULT_RENDER_START_FRAME = 0;
constexpr uint32_t DEFAULT_RENDER_END_FRAME = 120;
constexpr double DEFAULT_RENDER_TIME = 0.0;
const std::string DEFAULT_PLAYBACK_LOAD_FOLDER = UNSET_PATH;

// Display settings
constexpr bool DEFAULT_SHOW_PARTICLES = true;
constexpr bool DEFAULT_SHOW_BODIES = true;

// Resource paths - using std::string consistently
const std::string DEFAULT_TEXTURE_FOLDER = UNSET_PATH;
const std::string DEFAULT_GLTF_MODEL_FOLDER = UNSET_PATH;

const std::string DEFAULT_CUDA_GPU_ARCHITECTURES = std::string("sm_60,sm_75,sm_80,sm_86,sm_90");