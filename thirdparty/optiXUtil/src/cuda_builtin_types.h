#pragma once

/*

   Copyright 2023 Shin Watanabe

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

*/

//#include "platform_config.h"

// CUDA builtin type compatibility layer for host-side execution
//
// This header provides host-side definitions of CUDA's built-in vector types
// when compiling for CPU execution. It enables seamless code sharing between
// host and device code without requiring separate implementations.
//
// Key Features:
// - Conditional compilation: Only active when NOT compiling for CUDA device
// - Memory alignment compatibility: Matches CUDA's builtin type alignment
// - Complete type coverage: int2/3/4, uint2/3/4, float2/3/4 vector types
// - Constructor compatibility: Uniform and component-wise initialization
// - Array access operators: For flexible component manipulation
// - Make functions: Compatible with CUDA's make_* convention
//
// Supported Vector Types:
// - Integer vectors: int2, int3, int4 with 32-bit signed components
// - Unsigned integer vectors: uint2, uint3, uint4 with 32-bit unsigned components
// - Floating-point vectors: float2, float3, float4 with 32-bit float components
//
// Memory Alignment:
// - 2-component vectors: 8-byte aligned for optimal SIMD operations
// - 3-component vectors: 4-byte aligned (no padding to maintain compatibility)
// - 4-component vectors: 16-byte aligned for maximum vectorization efficiency
//
// Common Applications:
// - Cross-platform CUDA/CPU code development
// - Host-side vector operations matching GPU implementations
// - Testing and debugging GPU algorithms on CPU
// - Unified math libraries supporting both architectures
// - Data structure definitions shared between host and device
//
// Usage Pattern:
// Code written using these types can be compiled for both CPU and GPU
// without modification, enabling true write-once, run-anywhere vector
// mathematics for high-performance computing applications.


#if !defined(__CUDA_ARCH__) && !defined(__CUDACC__)

// 2-component signed integer vector with 8-byte alignment
struct alignas (8) int2
{
    int32_t x, y;
    // Uniform constructor initializes both components to the same value
    constexpr int2 (const int32_t v = 0) :
        x (v), y (v) {}
    // Component-wise constructor for explicit X and Y values
    constexpr int2 (const int32_t xx, const int32_t yy) :
        x (xx), y (yy) {}
};
// Factory function for creating int2 vectors (CUDA compatibility)
inline constexpr int2 make_int2 (const int32_t x, const int32_t y)
{
    return int2 (x, y);
}

// 3-component signed integer vector with 4-byte alignment
struct alignas (4) int3
{
    int32_t x, y, z;
    // Uniform constructor initializes all components to the same value
    constexpr int3 (const int32_t v = 0) :
        x (v), y (v), z (v) {}
    // Component-wise constructor for explicit X, Y, Z values
    constexpr int3 (const int32_t xx, const int32_t yy, const int32_t zz) :
        x (xx), y (yy), z (zz) {}
};
// Factory function for creating int3 vectors (CUDA compatibility)
inline constexpr int3 make_int3 (const int32_t x, const int32_t y, const int32_t z)
{
    return int3 (x, y, z);
}

// 4-component signed integer vector with 16-byte alignment for optimal SIMD
struct alignas (16) int4
{
    int32_t x, y, z, w;
    // Uniform constructor initializes all components to the same value
    constexpr int4 (const int32_t v = 0) :
        x (v), y (v), z (v), w (v) {}
    // Component-wise constructor for explicit X, Y, Z, W values
    constexpr int4 (const int32_t xx, const int32_t yy, const int32_t zz, const int32_t ww) :
        x (xx), y (yy), z (zz), w (ww) {}
};
// Factory function for creating int4 vectors (CUDA compatibility)
inline constexpr int4 make_int4 (const int32_t x, const int32_t y, const int32_t z, const int32_t w)
{
    return int4 (x, y, z, w);
}

// 2-component unsigned integer vector with 8-byte alignment
struct alignas (8) uint2
{
    uint32_t x, y;
    // Uniform constructor initializes both components to the same value
    constexpr uint2 (const uint32_t v = 0) :
        x (v), y (v) {}
    // Component-wise constructor for explicit X and Y values
    constexpr uint2 (const uint32_t xx, const uint32_t yy) :
        x (xx), y (yy) {}
    // Array-style access operator for mutable component access
    uint32_t& operator[] (uint32_t idx)
    {
        return *(&x + idx);
    }
    // Array-style access operator for const component access
    const uint32_t& operator[] (uint32_t idx) const
    {
        return *(&x + idx);
    }
};
// Factory function for creating uint2 vectors (CUDA compatibility)
inline constexpr uint2 make_uint2 (const uint32_t x, const uint32_t y)
{
    return uint2 (x, y);
}

// 3-component unsigned integer vector with 4-byte alignment
struct alignas (4) uint3
{
    uint32_t x, y, z;
    // Uniform constructor initializes all components to the same value
    constexpr uint3 (const uint32_t v = 0) :
        x (v), y (v), z (v) {}
    // Component-wise constructor for explicit X, Y, Z values
    constexpr uint3 (const uint32_t xx, const uint32_t yy, const uint32_t zz) :
        x (xx), y (yy), z (zz) {}
    // Array-style access operator for mutable component access
    uint32_t& operator[] (uint32_t idx)
    {
        return *(&x + idx);
    }
    // Array-style access operator for const component access
    const uint32_t& operator[] (uint32_t idx) const
    {
        return *(&x + idx);
    }
};
// Factory function for creating uint3 vectors (CUDA compatibility)
inline constexpr uint3 make_uint3 (const uint32_t x, const uint32_t y, const uint32_t z)
{
    return uint3 (x, y, z);
}

// 4-component unsigned integer vector with 16-byte alignment for optimal SIMD
struct alignas (16) uint4
{
    uint32_t x, y, z, w;
    // Uniform constructor initializes all components to the same value
    constexpr uint4 (const uint32_t v = 0) :
        x (v), y (v), z (v), w (v) {}
    // Component-wise constructor for explicit X, Y, Z, W values
    constexpr uint4 (const uint32_t xx, const uint32_t yy, const uint32_t zz, const uint32_t ww) :
        x (xx), y (yy), z (zz), w (ww) {}
    // Array-style access operator for mutable component access
    uint32_t& operator[] (uint32_t idx)
    {
        return *(&x + idx);
    }
    // Array-style access operator for const component access
    const uint32_t& operator[] (uint32_t idx) const
    {
        return *(&x + idx);
    }
};
// Factory function for creating uint4 vectors (CUDA compatibility)
inline constexpr uint4 make_uint4 (const uint32_t x, const uint32_t y, const uint32_t z, const uint32_t w)
{
    return uint4 (x, y, z, w);
}

// 2-component floating-point vector with 8-byte alignment
struct alignas (8) float2
{
    float x, y;
    // Uniform constructor initializes both components to the same value
    constexpr float2 (const float v = 0.0f) :
        x (v), y (v) {}
    // Component-wise constructor for explicit X and Y values
    constexpr float2 (const float xx, const float yy) :
        x (xx), y (yy) {}
    // Array-style access operator for mutable component access
    float& operator[] (uint32_t idx)
    {
        return *(&x + idx);
    }
    // Array-style access operator for const component access
    const float& operator[] (uint32_t idx) const
    {
        return *(&x + idx);
    }
};
// Factory function for creating float2 vectors (CUDA compatibility)
inline constexpr float2 make_float2 (const float x, const float y)
{
    return float2 (x, y);
}

// 3-component floating-point vector with 4-byte alignment
struct alignas (4) float3
{
    float x, y, z;
    // Uniform constructor initializes all components to the same value
    constexpr float3 (const float v = 0.0f) :
        x (v), y (v), z (v) {}
    // Component-wise constructor for explicit X, Y, Z values
    constexpr float3 (const float xx, const float yy, const float zz) :
        x (xx), y (yy), z (zz) {}
    // Array-style access operator for mutable component access
    float& operator[] (uint32_t idx)
    {
        return *(&x + idx);
    }
    // Array-style access operator for const component access
    const float& operator[] (uint32_t idx) const
    {
        return *(&x + idx);
    }
};
// Factory function for creating float3 vectors (CUDA compatibility)
inline constexpr float3 make_float3 (const float x, const float y, const float z)
{
    return float3 (x, y, z);
}

// 4-component floating-point vector with 16-byte alignment for optimal SIMD
struct alignas (16) float4
{
    float x, y, z, w;
    // Uniform constructor initializes all components to the same value
    constexpr float4 (const float v = 0.0f) :
        x (v), y (v), z (v), w (v) {}
    // Component-wise constructor for explicit X, Y, Z, W values
    constexpr float4 (const float xx, const float yy, const float zz, const float ww) :
        x (xx), y (yy), z (zz), w (ww) {}

    constexpr float4 (const float3& xyz, const float ww) :
        x (xyz.x), y (xyz.y), z (xyz.z), w (ww) {}

    // Array-style access operator for mutable component access
    float& operator[] (uint32_t idx)
    {
        return *(&x + idx);
    }
    // Array-style access operator for const component access
    const float& operator[] (uint32_t idx) const
    {
        return *(&x + idx);
    }
};
// Factory function for creating float4 vectors (CUDA compatibility)
inline constexpr float4 make_float4 (const float x, const float y, const float z, const float w)
{
    return float4 (x, y, z, w);
}

#endif // !defined(__CUDA_ARCH__) && !defined(__CUDACC__)