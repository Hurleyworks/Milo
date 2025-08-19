#pragma once

#include "common/common_host.h"

constexpr int INVALID_RAY_TYPE = -1;

enum class BufferToDisplay
{
    NoisyBeauty = 0,
    Albedo,
    Normal,
    Flow,
    DenoisedBeauty,
};

struct GAS
{
    optixu::GeometryAccelerationStructure gas;
    cudau::Buffer gasMem;
};
