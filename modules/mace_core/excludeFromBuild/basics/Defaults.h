#pragma once



// You can make it constexpr if its value is known at compile - time and won't change.
// constexpr ensures that the value is computed at compile-time.
constexpr uint32_t DEFAULT_GUI_HEADER_HEIGHT = 40; // pixels
constexpr uint32_t DEFAULT_GUI_FOOTER_HEIGHT = 40; // pixels
constexpr float PHI = 1.618f;
constexpr float DEFAULT_DESKTOP_WINDOW_HEIGHT = 800.0f + DEFAULT_GUI_HEADER_HEIGHT + DEFAULT_GUI_FOOTER_HEIGHT;
constexpr float DEFAULT_DESKTOP_WINDOW_WIDTH = DEFAULT_DESKTOP_WINDOW_HEIGHT * PHI;

const Eigen::Vector2i DEFAULT_DESKTOP_WINDOW_SIZE = Eigen::Vector2i (int (DEFAULT_DESKTOP_WINDOW_WIDTH), int (DEFAULT_DESKTOP_WINDOW_HEIGHT));
constexpr int DEFAULT_MIN_WINDOW = 10;

// std::string is not a literal type, so it can't be used with constexpr.
// constexpr requires the variable to be initialized with a constant expression,
// and std::string involves dynamic memory allocation.
const std::string DEFAULT_DESKTOP_WINDOW_NAME = "DesktopWindow";

constexpr int DEFAULT_DESKTOP_WINDOW_REFRESH_RATE = 16;
constexpr bool DEFAULT_DESKTOP_WINDOW_RESIZABLE = true;

constexpr float MAX_FLOAT = 3.402823466e+38f; // Approximate value of max float


struct GPUMemoryStats
{
    size_t usedMemory = 0;
    size_t freeMemory = 0;
    size_t totalMemory = 0;
    float usagePercent = 0.0f;
};


struct DesktopWindowSettings
{
    uint32_t width = static_cast<uint32_t> (DEFAULT_DESKTOP_WINDOW_WIDTH);
    uint32_t height = static_cast<uint32_t> (DEFAULT_DESKTOP_WINDOW_HEIGHT);
    std::string name = DEFAULT_DESKTOP_WINDOW_NAME;
    int refreshRate = DEFAULT_DESKTOP_WINDOW_REFRESH_RATE;
    bool resizable = DEFAULT_DESKTOP_WINDOW_RESIZABLE;
};

// mapped from GLFW
#define MOUSE_BUTTON_1 0
#define MOUSE_BUTTON_2 1
#define MOUSE_BUTTON_3 2
#define MOUSE_BUTTON_4 3
#define MOUSE_BUTTON_5 4
#define MOUSE_BUTTON_6 5
#define MOUSE_BUTTON_7 6
#define MOUSE_BUTTON_8 7
#define MOUSE_BUTTON_LAST MOUSE_BUTTON_8
#define MOUSE_BUTTON_LEFT MOUSE_BUTTON_1
#define MOUSE_BUTTON_RIGHT MOUSE_BUTTON_2
#define MOUSE_BUTTON_MIDDLE MOUSE_BUTTON_3

#define MOUSE_RELEASE 0
#define MOUSE_PRESS 1
#define MOUSE_REPEAT 2

using ItemID = int64_t;
using BodyID = int64_t;
using PolyID = int64_t;
constexpr int64_t INVALID_ID = -1;
constexpr int64_t INVALID_INDEX = -1;

const float MIN_SCALE_F = std::numeric_limits<float>::min();
const float MIN_SCALE_D = std::numeric_limits<double>::min();

//const std::string INVALID_PATH = "invalid path";
const std::string UNSET_PATH = "";
const std::string INVALID_NAME = "invalid name";
const std::string DEFAULT_ERROR_MESSAGE = "AOK";
const std::string DEFAULT_CAMERA_NAME = "Default_Camera";

using PathList = std::vector<std::filesystem::path>;

// particles
using ParticleData = std::vector<Eigen::Vector4f>;


