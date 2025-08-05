#pragma once

static const char* IOKeyTable[] =
    {
        "SceneIconCount",
        "ModelIconCount",
        "EnviroIconCount",
        "TextureIconCount",
        "Invalid"};

struct IOKey
{
    enum EIOKey
    {
        SceneIconCount,
        ModelIconCount,
        EnviroIconCount,
        TextureIconCount,
        Count,
        Invalid = Count
    };

    union
    {
        EIOKey name;
        unsigned int value;
    };

    IOKey (EIOKey name) :
        name (name) {}
    IOKey (unsigned int value) :
        value (value) {}
    IOKey() :
        value (Invalid) {}
    operator EIOKey() const { return name; }
    const char* ToString() const { return IOKeyTable[value]; }
    static IOKey FromString (const char* str) { return mace::TableLookup (str, IOKeyTable, Count); }
};

const uint32_t DEFAULT_ICON_COUNT = 10;

using IOProperties = AnyValue<IOKey>;
using IOPropsRef = std::shared_ptr<IOProperties>;
