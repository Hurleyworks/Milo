#pragma once

using sabi::PhysicsEngineState;
using sabi::BodyType;
using sabi::CollisionShape;

// Modern enum classes for physics-related types
enum class PhysicsSolverMode
{
    Avx2,
    Cuda,
    Count,
    Invalid = Count
};

enum class PhysicsEngineType
{
    Newton,
    Flex,
    Splash,
    Count,
    Invalid = Count
};

// Comprehensive enum class for physics property keys
enum class PhysicsKeys
{
    PhysicsEngineState,

    // Body properties
    CollisionShape,
    BodyType,
    Mass,
    StaticFriction,
    DynamicFriction,
    Bounciness,
    Force,
    ImpulseSpeed,

    Count,
    Invalid = Count
};

// String conversion tables - separated for cleaner organization
namespace EnumStrings
{
    static const char* PhysicsSolverModeStrings[] = {
        "Avx2", "Cuda", "Invalid"};

    static const char* PhysicsEngineTypeStrings[] = {
        "Newton", "Flex", "Splash", "Invalid"};

    static const char* PhysicsKeyStrings[] = {
        "PhysicsEngineState",
        "CollisionShape",
        "BodyType",
        "Mass",
        "StaticFriction",
        "DynamicFriction",
        "Bounciness",
        "Force",
        "ImpulseSpeed",
        "Invalid"};
} // namespace EnumStrings

// String conversion functions
inline const char* ToString (PhysicsSolverMode mode)
{
    return EnumStrings::PhysicsSolverModeStrings[static_cast<int> (mode) >= static_cast<int> (PhysicsSolverMode::Invalid) ? static_cast<int> (PhysicsSolverMode::Invalid) : static_cast<int> (mode)];
}

inline const char* ToString (PhysicsEngineType type)
{
    return EnumStrings::PhysicsEngineTypeStrings[static_cast<int> (type) >= static_cast<int> (PhysicsEngineType::Invalid) ? static_cast<int> (PhysicsEngineType::Invalid) : static_cast<int> (type)];
}

inline const char* ToString (PhysicsKeys key)
{
    return EnumStrings::PhysicsKeyStrings[static_cast<int> (key) >= static_cast<int> (PhysicsKeys::Invalid) ? static_cast<int> (PhysicsKeys::Invalid) : static_cast<int> (key)];
}

// String to enum conversion functions
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

inline PhysicsSolverMode PhysicsSolverModeFromString (const char* str)
{
    return FromString<PhysicsSolverMode> (str, EnumStrings::PhysicsSolverModeStrings,
                                          static_cast<int> (PhysicsSolverMode::Count));
}

inline PhysicsEngineType PhysicsEngineTypeFromString (const char* str)
{
    return FromString<PhysicsEngineType> (str, EnumStrings::PhysicsEngineTypeStrings,
                                          static_cast<int> (PhysicsEngineType::Count));
}

inline PhysicsKeys PhysicsKeyFromString (const char* str)
{
    return FromString<PhysicsKeys> (str, EnumStrings::PhysicsKeyStrings,
                                   static_cast<int> (PhysicsKeys::Count));
}

// Type definitions for physics property collection
using PhysicsProperties = AnyValue<PhysicsKeys>;
using PhysicsPropsRef = std::shared_ptr<PhysicsProperties>;

// Default constants
const PhysicsEngineState DEFAULT_PHYSICS_ENGINE_STATE = PhysicsEngineState::Pause;
const PhysicsEngineType DEFAULT_PHYSICS_ENGINE_TYPE = PhysicsEngineType::Newton;
const PhysicsSolverMode DEFAULT_PHYSICS_SOLVER_MODE = PhysicsSolverMode::Avx2;
constexpr double DEFAULT_PHYSICS_MASS = 2.0;
constexpr bool DEFAULT_GET_CONTACT_POINTS = false;
