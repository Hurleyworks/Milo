
#pragma once

static const char* WorldKeyTable[] =
    {
        "MinInstScale",
        "MaxInstScale",
        "Invalid"};

struct WorldKey
{
    enum EWorldKey
    {
        MinInstScale,
        MaxInstScale,
        Count,
        Invalid = Count
    };

    union
    {
        EWorldKey name;
        unsigned int value;
    };

    WorldKey (EWorldKey name) :
        name (name) {}
    WorldKey (unsigned int value) :
        value (value) {}
    WorldKey() :
        value (Invalid) {}
    operator EWorldKey() const { return name; }
    const char* ToString() const { return WorldKeyTable[value]; }
};

typedef AnyValue<WorldKey> WorldProperties;
using WorldPropsRef = std::shared_ptr<WorldProperties>;

const int32_t DEFAULT_MIN_INSTANCE_SCALE = 100;
const int32_t DEFAULT_MAX_INSTANCE_SCALE = 100;