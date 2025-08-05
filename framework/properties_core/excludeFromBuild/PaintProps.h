#pragma once

// Constant for invalid collision count
constexpr int INVALID_COLLISION_COUNT = -1;

// Modern enum classes for paint-related types
enum class AxisType
{
    X,
    Y,
    Z,
    Count,
    Invalid = Count
};

enum class Sense
{
    Positive,
    Negative,
    Count,
    Invalid = Count
};

enum class UpDirection
{
    PolyNormal,
    WorldY,
    Count,
    Invalid = Count
};

enum class EmitType
{
    Item,
    Mouse,
    Count,
    Invalid = Count
};

enum class PaintToolType
{
    Transform,
    Tweak,
    Paint,
    Emit,
    Grow,
    Wall,
    Impulse,
    Count,
    Invalid = Count
};

enum class TweakMode
{
    Transform,
    Replace,
    Delete,
    Count,
    Invalid = Count
};

enum class PaintMode
{
    Apply,
    Remove,
    Replace,
    Count,
    Invalid = Count
};

enum class PaintRemoveMode
{
    Brush,
    Box,
    Sphere,
    Count,
    Invalid = Count
};

enum class PaintToolState
{
    Active,
    Idle,
    Count,
    Invalid = Count
};

// Comprehensive enum class for paint property keys
enum class PaintKey
{
    toolType,
    toolState,
    centerOnSurface,
    axis,
    sense,
    offsetFromSurface,
    upDirection,
    alignToDrag,
    dragSense,
    dragSmoothing,
    minScale,
    maxScale,
    uniformScale,
    randomScale,
    minRotation,
    maxRotation,
    randomRotation,
    emitType,

    // tweak
    tweakMode,

    // paint
    paintMode,

    // grow
    growCount,
    growSpacing,

    // wall
    wallRows,
    wallColumns,
    wallHorizontalGap,
    wallVerticalGap,

    // impulse
    sendImpulseInstances,

    Count,
    Invalid = Count
};

// Type definitions for paint property collection
using PaintProperties = AnyValue<PaintKey>;
using PaintPropsRef = std::shared_ptr<PaintProperties>;

// Constants for default values
const PaintToolType DEFAULT_PAINT_TOOL = PaintToolType::Tweak;
const TweakMode DEFAULT_TWEAK_MODE = TweakMode::Transform;
const PaintMode DEFAULT_PAINT_MODE = PaintMode::Apply;
const EmitType DEFAULT_EMIT_TYPE = EmitType::Mouse;
const PaintToolState DEFAULT_PAINT_TOOL_STATE = PaintToolState::Idle;
constexpr int DEFAULT_CENTER_ON_SURFACE = 0;
const AxisType DEFAULT_AXIS_TYPE = AxisType::Y;
const Sense DEFAULT_SENSE = Sense::Negative;
constexpr double DEFAULT_OFFSET_FROM_SURFACE = 0.0;
const UpDirection DEFAULT_UP_DIRECTION = UpDirection::PolyNormal;
constexpr int DEFAULT_ALIGN_TO_DRAG = 1;
constexpr double DEFAULT_DRAG_SMOOTHING = 5.0;
const Sense DEFAULT_DRAG_SENSE = Sense::Positive;
const Eigen::Vector3d DEFAULT_MIN_SCALE = Eigen::Vector3d (1, 1, 1);
const Eigen::Vector3d DEFAULT_MAX_SCALE = Eigen::Vector3d (1, 1, 1);
constexpr int DEFAULT_UNIFORM_SCALE = 1;
constexpr int DEFAULT_RANDOM_SCALE = 1;
constexpr double DEFAULT_MIN_ROTATION = 0.0;
constexpr double DEFAULT_MAX_ROTATION = 0.0;
constexpr int DEFAULT_RANDOM_ROTATION = 1;

// grow
constexpr uint32_t DEFAULT_GROW_COUNT = 0;
constexpr double DEFAULT_GROW_SPACING = 0.0;

// wall
constexpr uint32_t DEFAULT_WALL_ROWS = 5;
constexpr uint32_t DEFAULT_WALL_COLS = 10;
constexpr double DEFAULT_WALL_HORIZONTAL_GAP = 0.1;
constexpr double DEFAULT_WALL_VERTICAL_GAP = 0.1;

// impulse
constexpr int DEFAULT_SEND_IMPULSE_INSTANCES = 0;

// String conversion utilities (separate implementation needed if required)
// These can be implemented as free functions if string conversion is needed:
// std::string AxisTypeToString(AxisType type);
// AxisType AxisTypeFromString(const std::string& str);
// etc.