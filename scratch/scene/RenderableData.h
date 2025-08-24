#pragma once

// Modern enum classes for physics and rendering entities
enum class PhysicsEngineState
{
    Start,
    Step,
    Pause,
    Reset,
    Count,
    Invalid = Count
};

enum class BodyType
{
    None,
    Static,
    Dynamic,
    Fluid,
    Count,
    Invalid = Count
};

enum class CollisionShape
{
    None,
    Ball,
    Box,
    ConvexHull,
    Mesh,
    Composite,
    Compound,
    Count,
    Invalid = Count
};

enum class EmitterType
{
    Zaxis,
    Vertices,
    None,
    Count,
    Invalid = Count
};

enum class EmitMode
{
    Mouse,
    Item,
    Count,
    Invalid = Count
};

enum class PrimitiveType
{
    Tetrahedron,
    Hexahedron,
    Octahedron,
    Dodecahedron,
    Icosahedron,
    Icosphere,
    Sphere,
    Plane,
    Cone,
    Cylinder,
    Capsule,
    Torus,
    Bowl,
    Count,
    Invalid = Count
};

// String conversion tables - separated from enum definitions for cleaner organization
namespace EnumStrings
{
    // Define string conversion tables
    static const char* PhysicsEngineStateStrings[] = {
        "Start", "Step", "Pause", "Reset", "Invalid"};

    static const char* BodyTypeStrings[] = {
        "None", "Static", "Dynamic", "Fluid", "Invalid"};

    static const char* CollisionShapeStrings[] = {
        "None", "Ball", "Box", "Convex hull", "Mesh", "Composite", "Compound", "Invalid"};

    static const char* EmitterTypeStrings[] = {
        "Zaxis", "Vertices", "None", "Invalid"};

    static const char* EmitModeStrings[] = {
        "Mouse", "Item", "Invalid"};

    static const char* PrimitiveTypeStrings[] = {
        "Tetrahedron", "Hexahedron", "Octahedron", "Dodecahedron", "Icosahedron",
        "Icosphere", "Sphere", "Plane", "Cone", "Cylinder", "Capsule", "Torus", "Bowl", "Invalid"};
} // namespace EnumStrings

// String conversion functions
inline const char* ToString (PhysicsEngineState state)
{
    return EnumStrings::PhysicsEngineStateStrings[static_cast<int> (state) >= static_cast<int> (PhysicsEngineState::Invalid) ? static_cast<int> (PhysicsEngineState::Invalid) : static_cast<int> (state)];
}

inline const char* ToString (BodyType type)
{
    return EnumStrings::BodyTypeStrings[static_cast<int> (type) >= static_cast<int> (BodyType::Invalid) ? static_cast<int> (BodyType::Invalid) : static_cast<int> (type)];
}

inline const char* ToString (CollisionShape shape)
{
    return EnumStrings::CollisionShapeStrings[static_cast<int> (shape) >= static_cast<int> (CollisionShape::Invalid) ? static_cast<int> (CollisionShape::Invalid) : static_cast<int> (shape)];
}

inline const char* ToString (EmitterType type)
{
    return EnumStrings::EmitterTypeStrings[static_cast<int> (type) >= static_cast<int> (EmitterType::Invalid) ? static_cast<int> (EmitterType::Invalid) : static_cast<int> (type)];
}

inline const char* ToString (EmitMode mode)
{
    return EnumStrings::EmitModeStrings[static_cast<int> (mode) >= static_cast<int> (EmitMode::Invalid) ? static_cast<int> (EmitMode::Invalid) : static_cast<int> (mode)];
}

inline const char* ToString (PrimitiveType type)
{
    return EnumStrings::PrimitiveTypeStrings[static_cast<int> (type) >= static_cast<int> (PrimitiveType::Invalid) ? static_cast<int> (PrimitiveType::Invalid) : static_cast<int> (type)];
}

// String to enum conversion function
template <typename EnumType>
EnumType FromString (const char* str, const char* const* stringArray, int count)
{
    for (int i = 0; i < count; i++)
    {
        if (strcmp (str, stringArray[i]) == 0)
        {
            return static_cast<EnumType> (i);
        }
    }
    return static_cast<EnumType> (count); // Return Invalid
}

// Specialized FromString functions
inline BodyType BodyTypeFromString (const char* str)
{
    return FromString<BodyType> (str, EnumStrings::BodyTypeStrings, static_cast<int> (BodyType::Count));
}

inline CollisionShape CollisionShapeFromString (const char* str)
{
    return FromString<CollisionShape> (str, EnumStrings::CollisionShapeStrings, static_cast<int> (CollisionShape::Count));
}

inline EmitMode EmitModeFromString (const char* str)
{
    return FromString<EmitMode> (str, EnumStrings::EmitModeStrings, static_cast<int> (EmitMode::Count));
}

inline PrimitiveType PrimitiveTypeFromString (const char* str)
{
    return FromString<PrimitiveType> (str, EnumStrings::PrimitiveTypeStrings, static_cast<int> (PrimitiveType::Count));
}

// Default constants - grouped by category
// Physics defaults
const BodyType DEFAULT_BODY_TYPE = BodyType::None;
const CollisionShape DEFAULT_COLLISION_SHAPE = CollisionShape::None;
const EmitterType DEFAULT_EMITTER_TYPE = EmitterType::Zaxis;

// Material properties defaults
constexpr double DEFAULT_ADHESION = 0.0;
constexpr double DEFAULT_STATIC_MASS = 0.0;
constexpr double DEFAULT_DYNAMIC_MASS = 2.0;
constexpr double DEFAULT_STATIC_FRICTION = 0.8;
constexpr double DEFAULT_DYNAMIC_FRICTION = 0.4;
constexpr double DEFAULT_BOUNCINESS = 0.0;

// Force and motion defaults
const Eigen::Vector3d DEFAULT_FORCE = Eigen::Vector3d (0.0, -10.0, 0.0);
constexpr double DEFAULT_IMPULSE_SPEED = 0.0;
const Eigen::Vector3d DEFAULT_IMPULSE_DIRECTION = Eigen::Vector3d (0.0, 0.0, 0.0);
constexpr uint32_t DEFAULT_SLEEP_STATE = 0;