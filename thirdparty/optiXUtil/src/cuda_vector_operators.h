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
//#include "cuda_math_utils.h"
//#include "cuda_builtin_types.h"

//
// CUDA Built-in Vector Type Operators and Utilities
//
// This header provides comprehensive operator overloads and utility functions for CUDA's
// built-in vector types (int2/3/4, uint2/3/4, float2/3/4). Essential for high-performance
// graphics programming where vector operations need to work seamlessly on both CPU and GPU.
//
// Key Features:
//   - Type-safe concepts to constrain templates to specific vector types
//   - Automatic type promotion in mixed-type operations (e.g., int + float = float)
//   - Comprehensive arithmetic operators (+, -, *, /, %) with scalar/vector variants
//   - Bitwise operators for integer vector types (<<, >>, &, |, ^)
//   - Assignment operators (+=, -=, *=, /=, etc.)
//   - Comparison operators (==, !=)
//   - Utility functions (length, min, max, dot product)
//   - Optimized division using reciprocal multiplication for floating-point types
//   - Factory functions with automatic type conversion and component broadcasting
//
// Design Philosophy:
//   - All operations are constexpr and inline for maximum performance
//   - CUDA_COMMON_FUNCTION ensures host/device compatibility
//   - Template metaprogramming handles type deduction and promotion automatically
//   - Follows standard mathematical conventions for vector operations
//
// Performance Considerations:
//   - Division by scalars uses reciprocal multiplication when possible
//   - Component-wise operations are fully unrolled at compile time
//   - Zero runtime overhead for type conversions and operator dispatch
//
// CUDA Compatible: All functions work in both host and device code
//

// Concepts for native 2D integer vectors (int2, uint2)
template <typename T>
concept NVec2I =
    // std::is_same_v<T, char2> || std::is_same_v<T, uchar2> ||
    // std::is_same_v<T, short2> || std::is_same_v<T, ushort2> ||
    std::is_same_v<T, int2> || std::is_same_v<T, uint2> /* ||
     std::is_same_v<T, longlong2> || std::is_same_v<T, ulonglong2>*/
    ;

// Concepts for native 2D vectors (integer and floating-point)
template <typename T>
concept NVec2 =
    NVec2I<T> ||
    std::is_same_v<T, float2> /* ||
     std::is_same_v<T, double2>*/
    ;

// Concepts for native 3D integer vectors (int3, uint3)
template <typename T>
concept NVec3I =
    // std::is_same_v<T, char3> || std::is_same_v<T, uchar3> ||
    // std::is_same_v<T, short3> || std::is_same_v<T, ushort3> ||
    std::is_same_v<T, int3> || std::is_same_v<T, uint3> /* ||
     std::is_same_v<T, longlong3> || std::is_same_v<T, ulonglong3>*/
    ;

// Concepts for native 3D vectors (integer and floating-point)
template <typename T>
concept NVec3 =
    NVec3I<T> ||
    std::is_same_v<T, float3> /* ||
     std::is_same_v<T, double3>*/
    ;

// Concepts for native 4D integer vectors (int4, uint4)
template <typename T>
concept NVec4I =
    // std::is_same_v<T, char4> || std::is_same_v<T, uchar4> ||
    // std::is_same_v<T, short4> || std::is_same_v<T, ushort4> ||
    std::is_same_v<T, int4> || std::is_same_v<T, uint4> /* ||
     std::is_same_v<T, longlong4> || std::is_same_v<T, ulonglong4>*/
    ;

// Concepts for native 4D vectors (integer and floating-point)
template <typename T>
concept NVec4 =
    NVec4I<T> ||
    std::is_same_v<T, float4> /* ||
     std::is_same_v<T, double4>*/
    ;

// Template metaprogramming structure to get appropriate 2D vector type for given numeric type
template <Number N>
struct GetNVec2;
template <>
struct GetNVec2<int32_t>
{
    using Type = int2;
    // Factory function to create int2 from components
    CUDA_COMMON_FUNCTION CUDA_INLINE static Type make (
        const int32_t x, const int32_t y)
    {
        return make_int2 (x, y);
    }
};
template <>
struct GetNVec2<uint32_t>
{
    using Type = uint2;
    // Factory function to create uint2 from components
    CUDA_COMMON_FUNCTION CUDA_INLINE static Type make (
        const uint32_t x, const uint32_t y)
    {
        return make_uint2 (x, y);
    }
};
template <>
struct GetNVec2<float>
{
    using Type = float2;
    // Factory function to create float2 from components
    CUDA_COMMON_FUNCTION CUDA_INLINE static Type make (
        const float x, const float y)
    {
        return make_float2 (x, y);
    }
};
template <Number N>
using NVec2_t = typename GetNVec2<N>::Type;

// Template metaprogramming structure to get appropriate 3D vector type for given numeric type
template <Number N>
struct GetNVec3;
template <>
struct GetNVec3<int32_t>
{
    using Type = int3;
    // Factory function to create int3 from components
    CUDA_COMMON_FUNCTION CUDA_INLINE static Type make (
        const int32_t x, const int32_t y, const int32_t z)
    {
        return make_int3 (x, y, z);
    }
};
template <>
struct GetNVec3<uint32_t>
{
    using Type = uint3;
    // Factory function to create uint3 from components
    CUDA_COMMON_FUNCTION CUDA_INLINE static Type make (
        const uint32_t x, const uint32_t y, const uint32_t z)
    {
        return make_uint3 (x, y, z);
    }
};
template <>
struct GetNVec3<float>
{
    using Type = float3;
    // Factory function to create float3 from components
    CUDA_COMMON_FUNCTION CUDA_INLINE static Type make (
        const float x, const float y, const float z)
    {
        return make_float3 (x, y, z);
    }
};
template <Number N>
using NVec3_t = typename GetNVec3<N>::Type;

// Template metaprogramming structure to get appropriate 4D vector type for given numeric type
template <Number N>
struct GetNVec4;
template <>
struct GetNVec4<int32_t>
{
    using Type = int4;
    // Factory function to create int4 from components
    CUDA_COMMON_FUNCTION CUDA_INLINE static Type make (
        const int32_t x, const int32_t y, const int32_t z, const int32_t w)
    {
        return make_int4 (x, y, z, w);
    }
};
template <>
struct GetNVec4<uint32_t>
{
    using Type = uint4;
    // Factory function to create uint4 from components
    CUDA_COMMON_FUNCTION CUDA_INLINE static Type make (
        const uint32_t x, const uint32_t y, const uint32_t z, const uint32_t w)
    {
        return make_uint4 (x, y, z, w);
    }
};
template <>
struct GetNVec4<float>
{
    using Type = float4;
    // Factory function to create float4 from components
    CUDA_COMMON_FUNCTION CUDA_INLINE static Type make (
        const float x, const float y, const float z, const float w)
    {
        return make_float4 (x, y, z, w);
    }
};
template <Number N>
using NVec4_t = typename GetNVec4<N>::Type;

// Type promotion rules for binary operations between numeric types
template <Number NA, Number NB>
struct GetBinOpResultType;
template <>
struct GetBinOpResultType<int32_t, int32_t>
{
    using Type = int32_t;
};
template <>
struct GetBinOpResultType<int32_t, uint32_t>
{
    using Type = uint32_t; // int + uint = uint
};
template <>
struct GetBinOpResultType<int32_t, float>
{
    using Type = float; // int + float = float
};
template <>
struct GetBinOpResultType<uint32_t, int32_t>
{
    using Type = uint32_t; // uint + int = uint
};
template <>
struct GetBinOpResultType<uint32_t, uint32_t>
{
    using Type = uint32_t;
};
template <>
struct GetBinOpResultType<uint32_t, float>
{
    using Type = float; // uint + float = float
};
template <>
struct GetBinOpResultType<float, int32_t>
{
    using Type = float; // float + int = float
};
template <>
struct GetBinOpResultType<float, uint32_t>
{
    using Type = float; // float + uint = float
};
template <>
struct GetBinOpResultType<float, float>
{
    using Type = float;
};
template <Number NA, Number NB>
using GetBinOpResultType_t = typename GetBinOpResultType<NA, NB>::Type;

// Traits for binary operations between 2D vectors - handles type promotion and result construction
template <NVec2 NV2A, NVec2 NV2B>
struct Vec2BinaryOpTraits
{
    using UType = GetBinOpResultType_t<decltype (NV2A::x), decltype (NV2B::x)>;
    using ReturnType = NVec2_t<UType>;
    // Factory function to create result vector from promoted component type
    CUDA_COMMON_FUNCTION CUDA_INLINE static ReturnType make (
        const UType x, const UType y)
    {
        return GetNVec2<UType>::make (x, y);
    }
};

// Traits for binary operations between 3D vectors - handles type promotion and result construction
template <NVec3 NV3A, NVec3 NV3B>
struct Vec3BinaryOpTraits
{
    using UType = GetBinOpResultType_t<decltype (NV3A::x), decltype (NV3B::x)>;
    using ReturnType = NVec3_t<UType>;
    // Factory function to create result vector from promoted component type
    CUDA_COMMON_FUNCTION CUDA_INLINE static ReturnType make (
        const UType x, const UType y, const UType z)
    {
        return GetNVec3<UType>::make (x, y, z);
    }
};

// Traits for binary operations between 4D vectors - handles type promotion and result construction
template <NVec4 NV4A, NVec4 NV4B>
struct Vec4BinaryOpTraits
{
    using UType = GetBinOpResultType_t<decltype (NV4A::x), decltype (NV4B::x)>;
    using ReturnType = NVec4_t<UType>;
    // Factory function to create result vector from promoted component type
    CUDA_COMMON_FUNCTION CUDA_INLINE static ReturnType make (
        const UType x, const UType y, const UType z, const UType w)
    {
        return GetNVec4<UType>::make (x, y, z, w);
    }
};

// Enhanced make_int2 functions with automatic type conversion and broadcasting
// Create int2 from single value (broadcasts to both components)
template <Number N>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr int2 make_int2 (const N x)
{
    return ::make_int2 (x, x);
}

// Create int2 from two values with automatic type conversion
template <Number N, std::enable_if_t<!std::is_same_v<N, int32_t>, int> = 0>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr int2 make_int2 (const N x, const N y)
{
    return ::make_int2 (
        static_cast<int32_t> (x),
        static_cast<int32_t> (y));
}

// Create int2 from existing 2D vector (copy x,y components)
template <NVec2 N2>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr int2 make_int2 (const N2& v)
{
    return ::make_int2 (v.x, v.y);
}

// Create int2 from 3D vector (take x,y components)
template <NVec3 N3>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr int2 make_int2 (const N3& v)
{
    return ::make_int2 (v.x, v.y);
}

// Create int2 from 4D vector (take x,y components)
template <NVec4 N4>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr int2 make_int2 (const N4& v)
{
    return ::make_int2 (v.x, v.y);
}

// Enhanced make_uint2 functions with automatic type conversion and broadcasting
// Create uint2 from single value (broadcasts to both components)
template <Number N>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr uint2 make_uint2 (const N x)
{
    return ::make_uint2 (x, x);
}

// Create uint2 from two values with automatic type conversion
template <Number N, std::enable_if_t<!std::is_same_v<N, uint32_t>, int> = 0>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr uint2 make_uint2 (const N x, const N y)
{
    return ::make_uint2 (
        static_cast<uint32_t> (x),
        static_cast<uint32_t> (y));
}

// Create uint2 from existing 2D vector (copy x,y components)
template <NVec2 N2>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr uint2 make_uint2 (const N2& v)
{
    return ::make_uint2 (v.x, v.y);
}

// Create uint2 from 3D vector (take x,y components)
template <NVec3 N3>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr uint2 make_uint2 (const N3& v)
{
    return ::make_uint2 (v.x, v.y);
}

// Create uint2 from 4D vector (take x,y components)
template <NVec4 N4>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr uint2 make_uint2 (const N4& v)
{
    return ::make_uint2 (v.x, v.y);
}

// Enhanced make_float2 functions with automatic type conversion and broadcasting
// Create float2 from single value (broadcasts to both components)
template <Number N>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr float2 make_float2 (const N x)
{
    return ::make_float2 (x, x);
}

// Create float2 from two values with automatic type conversion
template <Number N, std::enable_if_t<!std::is_same_v<N, float>, int> = 0>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr float2 make_float2 (const N x, const N y)
{
    return ::make_float2 (
        static_cast<float> (x),
        static_cast<float> (y));
}

// Create float2 from existing 2D vector (copy x,y components)
template <NVec2 N2>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr float2 make_float2 (const N2& v)
{
    return ::make_float2 (v.x, v.y);
}

// Create float2 from 3D vector (take x,y components)
template <NVec3 N3>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr float2 make_float2 (const N3& v)
{
    return ::make_float2 (v.x, v.y);
}

// Create float2 from 4D vector (take x,y components)
template <NVec4 N4>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr float2 make_float2 (const N4& v)
{
    return ::make_float2 (v.x, v.y);
}

// Enhanced make_int3 functions with automatic type conversion and broadcasting
// Create int3 from single value (broadcasts to all components)
template <Number N>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr int3 make_int3 (const N x)
{
    return ::make_int3 (x, x, x);
}

// Create int3 from three values with automatic type conversion
template <Number N, std::enable_if_t<!std::is_same_v<N, int32_t>, int> = 0>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr int3 make_int3 (const N x, const N y, const N z)
{
    return ::make_int3 (
        static_cast<int32_t> (x),
        static_cast<int32_t> (y),
        static_cast<int32_t> (z));
}

// Create int3 from existing 3D vector (copy all components)
template <NVec3 N3>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr int3 make_int3 (const N3& v)
{
    return ::make_int3 (v.x, v.y, v.z);
}

// Create int3 from 4D vector (take x,y,z components)
template <NVec4 N4>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr int3 make_int3 (const N4& v)
{
    return ::make_int3 (v.x, v.y, v.z);
}

// Enhanced make_uint3 functions with automatic type conversion and broadcasting
// Create uint3 from single value (broadcasts to all components)
template <Number N>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr uint3 make_uint3 (const N x)
{
    return ::make_uint3 (x, x, x);
}

// Create uint3 from three values with automatic type conversion
template <Number N, std::enable_if_t<!std::is_same_v<N, uint32_t>, int> = 0>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr uint3 make_uint3 (const N x, const N y, const N z)
{
    return ::make_uint3 (
        static_cast<uint32_t> (x),
        static_cast<uint32_t> (y),
        static_cast<uint32_t> (z));
}

// Create uint3 from existing 3D vector (copy all components)
template <NVec3 N3>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr uint3 make_uint3 (const N3& v)
{
    return ::make_uint3 (v.x, v.y, v.z);
}

// Create uint3 from 4D vector (take x,y,z components)
template <NVec4 N4>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr uint3 make_uint3 (const N4& v)
{
    return ::make_uint3 (v.x, v.y, v.z);
}

// Enhanced make_float3 functions with automatic type conversion and broadcasting
// Create float3 from single value (broadcasts to all components)
template <Number N>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr float3 make_float3 (const N x)
{
    return ::make_float3 (x, x, x);
}

// Create float3 from three values with automatic type conversion
template <Number N, std::enable_if_t<!std::is_same_v<N, float>, int> = 0>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr float3 make_float3 (const N x, const N y, const N z)
{
    return ::make_float3 (
        static_cast<float> (x),
        static_cast<float> (y),
        static_cast<float> (z));
}

// Create float3 from existing 3D vector (copy all components)
template <NVec3 N3>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr float3 make_float3 (const N3& v)
{
    return ::make_float3 (v.x, v.y, v.z);
}

// Create float3 from 4D vector (take x,y,z components)
template <NVec4 N4>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr float3 make_float3 (const N4& v)
{
    return ::make_float3 (v.x, v.y, v.z);
}

// Enhanced make_int4 functions with automatic type conversion and broadcasting
// Create int4 from single value (broadcasts to all components except w=1)
template <Number N>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr int4 make_int4 (const N x)
{
    return ::make_int4 (x, x, x);
}

// Create int4 from four values with automatic type conversion
template <Number N, std::enable_if_t<!std::is_same_v<N, int32_t>, int> = 0>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr int4 make_int4 (const N x, const N y, const N z, const N w)
{
    return ::make_int4 (
        static_cast<int32_t> (x),
        static_cast<int32_t> (y),
        static_cast<int32_t> (z),
        static_cast<int32_t> (w));
}

// Create int4 from 3D vector plus w component
template <NVec3 N3, Number N>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr int4 make_int4 (const N3& v, const N w)
{
    return ::make_int4 (v.x, v.y, v.z, w);
}

// Create int4 from existing 4D vector (copy all components)
template <NVec4 N4>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr int4 make_int4 (const N4& v)
{
    return ::make_int4 (v.x, v.y, v.z);
}

// Enhanced make_uint4 functions with automatic type conversion and broadcasting
// Create uint4 from single value (broadcasts to all components except w=1)
template <Number N>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr uint4 make_uint4 (const N x)
{
    return ::make_uint4 (x, x, x);
}

// Create uint4 from four values with automatic type conversion
template <Number N, std::enable_if_t<!std::is_same_v<N, uint32_t>, int> = 0>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr uint4 make_uint4 (const N x, const N y, const N z, const N w)
{
    return ::make_uint4 (
        static_cast<uint32_t> (x),
        static_cast<uint32_t> (y),
        static_cast<uint32_t> (z),
        static_cast<uint32_t> (w));
}

// Create uint4 from 3D vector plus w component
template <NVec3 N3, Number N>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr uint4 make_uint4 (const N3& v, const N w)
{
    return ::make_uint4 (v.x, v.y, v.z, w);
}

// Create uint4 from existing 4D vector (copy all components)
template <NVec4 N4>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr uint4 make_uint4 (const N4& v)
{
    return ::make_uint4 (v.x, v.y, v.z);
}

// Enhanced make_float4 functions with automatic type conversion and broadcasting
// Create float4 from single value (broadcasts to all components except w=1)
template <Number N>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr float4 make_float4 (const N x)
{
    return ::make_float4 (x, x, x);
}

// Create float4 from four values with automatic type conversion
template <Number N, std::enable_if_t<!std::is_same_v<N, float>, int> = 0>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr float4 make_float4 (const N x, const N y, const N z, const N w)
{
    return ::make_float4 (
        static_cast<float> (x),
        static_cast<float> (y),
        static_cast<float> (z),
        static_cast<float> (w));
}

// Create float4 from 3D vector plus w component
template <NVec3 N3, Number N>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr float4 make_float4 (const N3& v, const N w)
{
    return ::make_float4 (v.x, v.y, v.z, w);
}

// Create float4 from existing 4D vector (copy all components)
template <NVec4 N4>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr float4 make_float4 (const N4& v)
{
    return ::make_float4 (v.x, v.y, v.z);
}

// 2D Vector Operators

// Unary plus operator for 2D vectors (returns copy)
template <NVec2 N2>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr N2 operator+ (const N2& v)
{
    return v;
}

// Unary minus operator for 2D vectors (component-wise negation)
template <NVec2 N2>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr N2 operator- (const N2& v)
{
    return GetNVec2<decltype (N2::x)>::make (-v.x, -v.y);
}

// Equality comparison for 2D vectors
template <NVec2 N2A, NVec2 N2B>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr bool operator== (
    const N2A& a, const N2B& b)
{
    return a.x == b.x && a.y == b.y;
}

// Inequality comparison for 2D vectors
template <NVec2 N2A, NVec2 N2B>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr bool operator!= (
    const N2A& a, const N2B& b)
{
    return a.x != b.x || a.y != b.y;
}

// Addition: scalar + vector (broadcasts scalar to both components)
template <Number NA, NVec2 N2B>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr typename Vec2BinaryOpTraits<NVec2_t<NA>, N2B>::ReturnType operator+ (
    const NA a, const N2B& b)
{
    return Vec2BinaryOpTraits<NVec2_t<NA>, N2B>::make (a + b.x, a + b.y);
}

// Addition: vector + scalar (broadcasts scalar to both components)
template <NVec2 N2A, Number NB>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr typename Vec2BinaryOpTraits<N2A, NVec2_t<NB>>::ReturnType operator+ (
    const N2A& a, const NB b)
{
    return Vec2BinaryOpTraits<N2A, NVec2_t<NB>>::make (a.x + b, a.y + b);
}

// Addition: vector + vector (component-wise)
template <NVec2 N2A, NVec2 N2B>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr typename Vec2BinaryOpTraits<N2A, N2B>::ReturnType operator+ (
    const N2A& a, const N2B& b)
{
    return Vec2BinaryOpTraits<N2A, N2B>::make (a.x + b.x, a.y + b.y);
}

// Subtraction: scalar - vector (broadcasts scalar to both components)
template <Number NA, NVec2 N2B>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr typename Vec2BinaryOpTraits<NVec2_t<NA>, N2B>::ReturnType operator- (
    const NA a, const N2B& b)
{
    return Vec2BinaryOpTraits<NVec2_t<NA>, N2B>::make (a - b.x, a - b.y);
}

// Subtraction: vector - scalar (broadcasts scalar to both components)
template <NVec2 N2A, Number NB>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr typename Vec2BinaryOpTraits<N2A, NVec2_t<NB>>::ReturnType operator- (
    const N2A& a, const NB b)
{
    return Vec2BinaryOpTraits<N2A, NVec2_t<NB>>::make (a.x - b, a.y - b);
}

// Subtraction: vector - vector (component-wise)
template <NVec2 N2A, NVec2 N2B>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr typename Vec2BinaryOpTraits<N2A, N2B>::ReturnType operator- (
    const N2A& a, const N2B& b)
{
    return Vec2BinaryOpTraits<N2A, N2B>::make (a.x - b.x, a.y - b.y);
}

// Multiplication: scalar * vector (broadcasts scalar to both components)
template <Number NA, NVec2 N2B>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr typename Vec2BinaryOpTraits<NVec2_t<NA>, N2B>::ReturnType operator* (
    const NA a, const N2B& b)
{
    return Vec2BinaryOpTraits<NVec2_t<NA>, N2B>::make (a * b.x, a * b.y);
}

// Multiplication: vector * scalar (broadcasts scalar to both components)
template <NVec2 N2A, Number NB>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr typename Vec2BinaryOpTraits<N2A, NVec2_t<NB>>::ReturnType operator* (
    const N2A& a, const NB b)
{
    return Vec2BinaryOpTraits<N2A, NVec2_t<NB>>::make (a.x * b, a.y * b);
}

// Multiplication: vector * vector (component-wise, Hadamard product)
template <NVec2 N2A, NVec2 N2B>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr typename Vec2BinaryOpTraits<N2A, N2B>::ReturnType operator* (
    const N2A& a, const N2B& b)
{
    return Vec2BinaryOpTraits<N2A, N2B>::make (a.x * b.x, a.y * b.y);
}

// Division: vector / scalar (optimized with reciprocal multiplication for floats)
template <NVec2 N2A, Number NB>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr typename Vec2BinaryOpTraits<N2A, NVec2_t<NB>>::ReturnType operator/ (
    const N2A& a, const NB b)
{
    if constexpr (std::is_floating_point_v<NB>)
    {
        const NB rb = static_cast<NB> (1) / b;
        return Vec2BinaryOpTraits<N2A, NVec2_t<NB>>::make (a.x * rb, a.y * rb);
    }
    else
    {
        return Vec2BinaryOpTraits<N2A, NVec2_t<NB>>::make (a.x / b, a.y / b);
    }
}

// Division: vector / vector (component-wise)
template <NVec2 N2A, NVec2 N2B>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr typename Vec2BinaryOpTraits<N2A, N2B>::ReturnType operator/ (
    const N2A& a, const N2B& b)
{
    return Vec2BinaryOpTraits<N2A, N2B>::make (a.x / b.x, a.y / b.y);
}

// Modulo: vector % scalar (integer vectors only)
template <NVec2I N2A, std::integral NB>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr typename Vec2BinaryOpTraits<N2A, NVec2_t<NB>>::ReturnType operator% (
    const N2A& a, const NB b)
{
    return Vec2BinaryOpTraits<N2A, NVec2_t<NB>>::make (a.x % b, a.y % b);
}

// Modulo: vector % vector (component-wise, integer vectors only)
template <NVec2I N2A, NVec2I N2B>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr typename Vec2BinaryOpTraits<N2A, N2B>::ReturnType operator% (
    const N2A& a, const N2B& b)
{
    return Vec2BinaryOpTraits<N2A, N2B>::make (a.x % b.x, a.y % b.y);
}

// Left shift: scalar << vector (integer types only)
template <std::integral NA, NVec2I N2B>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr typename Vec2BinaryOpTraits<NVec2_t<NA>, N2B>::ReturnType operator<< (
    const NA a, const N2B& b)
{
    return Vec2BinaryOpTraits<NVec2_t<NA>, N2B>::make (a << b.x, a << b.y);
}

// Left shift: vector << scalar (integer types only)
template <NVec2I N2A, std::integral NB>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr typename Vec2BinaryOpTraits<N2A, NVec2_t<NB>>::ReturnType operator<< (
    const N2A& a, const NB b)
{
    return Vec2BinaryOpTraits<N2A, NVec2_t<NB>>::make (a.x << b, a.y << b);
}

// Left shift: vector << vector (component-wise, integer types only)
template <NVec2I N2A, NVec2I N2B>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr typename Vec2BinaryOpTraits<N2A, N2B>::ReturnType operator<< (
    const N2A& a, const N2B& b)
{
    return Vec2BinaryOpTraits<N2A, N2B>::make (a.x << b.x, a.y << b.y);
}

// Right shift: scalar >> vector (integer types only)
template <std::integral NA, NVec2I N2B>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr typename Vec2BinaryOpTraits<NVec2_t<NA>, N2B>::ReturnType operator>> (
    const NA a, const N2B& b)
{
    return Vec2BinaryOpTraits<NVec2_t<NA>, N2B>::make (a >> b.x, a >> b.y);
}

// Right shift: vector >> scalar (integer types only)
template <NVec2I N2A, std::integral NB>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr typename Vec2BinaryOpTraits<N2A, NVec2_t<NB>>::ReturnType operator>> (
    const N2A& a, const NB b)
{
    return Vec2BinaryOpTraits<N2A, NVec2_t<NB>>::make (a.x >> b, a.y >> b);
}

// Right shift: vector >> vector (component-wise, integer types only)
template <NVec2I N2A, NVec2I N2B>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr typename Vec2BinaryOpTraits<N2A, N2B>::ReturnType operator>> (
    const N2A& a, const N2B& b)
{
    return Vec2BinaryOpTraits<N2A, N2B>::make (a.x >> b.x, a.y >> b.y);
}

// Compound assignment operators for 2D vectors

// Addition assignment: vector += vector
template <NVec2 N2A, NVec2 N2B>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr void operator+= (
    N2A& a, const N2B& b)
{
    a.x += b.x;
    a.y += b.y;
}

// Subtraction assignment: vector -= vector
template <NVec2 N2A, NVec2 N2B>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr void operator-= (
    N2A& a, const N2B& b)
{
    a.x -= b.x;
    a.y -= b.y;
}

// Multiplication assignment: vector *= scalar
template <NVec2 N2A, Number NB>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr void operator*= (
    N2A& a, const NB& b)
{
    a.x *= b;
    a.y *= b;
}

// Multiplication assignment: vector *= vector (component-wise)
template <NVec2 N2A, NVec2 N2B>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr void operator*= (
    N2A& a, const N2B& b)
{
    a.x *= b.x;
    a.y *= b.y;
}

// Division assignment: vector /= scalar (optimized for floating-point)
template <NVec2 N2A, Number NB>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr void operator/= (
    N2A& a, const NB& b)
{
    if constexpr (std::is_floating_point_v<NB>)
    {
        const NB rb = static_cast<NB> (1) / b;
        a.x *= rb;
        a.y *= rb;
    }
    else
    {
        a.x /= b;
        a.y /= b;
    }
}

// Division assignment: vector /= vector (component-wise)
template <NVec2 N2A, NVec2 N2B>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr void operator/= (
    N2A& a, const N2B& b)
{
    a.x /= b.x;
    a.y /= b.y;
}

// Modulo assignment: vector %= scalar (integer vectors only)
template <NVec2I N2A, std::integral NB>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr void operator%= (
    N2A& a, const NB& b)
{
    a.x %= b;
    a.y %= b;
}

// Modulo assignment: vector %= vector (component-wise, integer vectors only)
template <NVec2I N2A, NVec2I N2B>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr void operator%= (
    N2A& a, const N2B& b)
{
    a.x %= b.x;
    a.y %= b.y;
}

// Left shift assignment: vector <<= scalar (integer vectors only)
template <NVec2I N2A, std::integral NB>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr void operator<<= (
    N2A& a, const NB& b)
{
    a.x <<= b;
    a.y <<= b;
}

// Left shift assignment: vector <<= vector (component-wise, integer vectors only)
template <NVec2I N2A, NVec2I N2B>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr void operator<<= (
    N2A& a, const N2B& b)
{
    a.x <<= b.x;
    a.y <<= b.y;
}

// Right shift assignment: vector >>= scalar (integer vectors only)
template <NVec2I N2A, std::integral NB>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr void operator>>= (
    N2A& a, const NB& b)
{
    a.x >>= b;
    a.y >>= b;
}

// Right shift assignment: vector >>= vector (component-wise, integer vectors only)
template <NVec2I N2A, NVec2I N2B>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr void operator>>= (
    N2A& a, const N2B& b)
{
    a.x >>= b.x;
    a.y >>= b.y;
}

// Calculate Euclidean length/magnitude of 2D vector
template <NVec2 N2>
CUDA_COMMON_FUNCTION CUDA_INLINE decltype (N2::x) length (const N2& v)
{
    return std::sqrt (pow2 (v.x) + pow2 (v.y));
}

// Component-wise minimum: scalar and vector
template <Number NA, NVec4 N2B>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr typename Vec2BinaryOpTraits<NVec2_t<NA>, N2B>::ReturnType min (
    const NA a, const N2B& b)
{
    using UType = GetBinOpResultType_t<NA, decltype (N2B::x)>;
    return Vec2BinaryOpTraits<NVec2_t<NA>, N2B>::make (
        stc::min<UType> (a, b.x), stc::min<UType> (a, b.y));
}

// Component-wise minimum: vector and scalar
template <NVec2 N2A, Number NB>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr typename Vec2BinaryOpTraits<N2A, NVec2_t<NB>>::ReturnType min (
    const N2A& a, const NB b)
{
    using UType = GetBinOpResultType_t<decltype (N2A::x), NB>;
    return Vec2BinaryOpTraits<N2A, NVec2_t<NB>>::make (
        stc::min<UType> (a.x, b), stc::min<UType> (a.y, b));
}

// Component-wise minimum: vector and vector
template <NVec2 N2A, NVec2 N2B>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr typename Vec2BinaryOpTraits<N2A, N2B>::ReturnType min (
    const N2A& a, const N2B& b)
{
    using UType = GetBinOpResultType_t<decltype (N2A::x), decltype (N2B::x)>;
    return Vec2BinaryOpTraits<N2A, N2B>::make (stc::min<UType> (a.x, b.x), stc::min<UType> (a.y, b.y));
}

// Component-wise maximum: scalar and vector
template <Number NA, NVec4 N2B>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr typename Vec2BinaryOpTraits<NVec2_t<NA>, N2B>::ReturnType max (
    const NA a, const N2B& b)
{
    using UType = GetBinOpResultType_t<NA, decltype (N2B::x)>;
    return Vec2BinaryOpTraits<NVec2_t<NA>, N2B>::make (
        stc::max<UType> (a, b.x), stc::max<UType> (a, b.y));
}

// Component-wise maximum: vector and scalar
template <NVec2 N2A, Number NB>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr typename Vec2BinaryOpTraits<N2A, NVec2_t<NB>>::ReturnType max (
    const N2A& a, const NB b)
{
    using UType = GetBinOpResultType_t<decltype (N2A::x), NB>;
    return Vec2BinaryOpTraits<N2A, NVec2_t<NB>>::make (
        stc::max<UType> (a.x, b), stc::max<UType> (a.y, b));
}

// Component-wise maximum: vector and vector
template <NVec2 N2A, NVec2 N2B>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr typename Vec2BinaryOpTraits<N2A, N2B>::ReturnType max (
    const N2A& a, const N2B& b)
{
    using UType = GetBinOpResultType_t<decltype (N2A::x), decltype (N2B::x)>;
    return Vec2BinaryOpTraits<N2A, N2B>::make (stc::max<UType> (a.x, b.x), stc::max<UType> (a.y, b.y));
}

// Dot product of two 2D vectors
template <NVec2 N2A, NVec2 N2B>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr GetBinOpResultType_t<decltype (N2A::x), decltype (N2B::x)> dot (
    const N2A& a, const N2B& b)
{
    return a.x * b.x + a.y * b.y;
}

// 3D Vector Operators

// Unary plus operator for 3D vectors (returns copy)
template <NVec3 N3>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr N3 operator+ (const N3& v)
{
    return v;
}

// Unary minus operator for 3D vectors (component-wise negation)
template <NVec3 N3>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr N3 operator- (const N3& v)
{
    return GetNVec3<decltype (N3::x)>::make (-v.x, -v.y, -v.z);
}

// Equality comparison for 3D vectors
template <NVec3 N3A, NVec3 N3B>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr bool operator== (
    const N3A& a, const N3B& b)
{
    return a.x == b.x && a.y == b.y && a.z == b.z;
}

// Inequality comparison for 3D vectors
template <NVec3 N3A, NVec3 N3B>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr bool operator!= (
    const N3A& a, const N3B& b)
{
    return a.x != b.x || a.y != b.y || a.z != b.z;
}

// Addition: scalar + vector (broadcasts scalar to all components)
template <Number NA, NVec3 N3B>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr typename Vec3BinaryOpTraits<NVec3_t<NA>, N3B>::ReturnType operator+ (
    const NA a, const N3B& b)
{
    return Vec3BinaryOpTraits<NVec3_t<NA>, N3B>::make (a + b.x, a + b.y, a + b.z);
}

// Addition: vector + scalar (broadcasts scalar to all components)
template <NVec3 N3A, Number NB>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr typename Vec3BinaryOpTraits<N3A, NVec3_t<NB>>::ReturnType operator+ (
    const N3A& a, const NB b)
{
    return Vec3BinaryOpTraits<N3A, NVec3_t<NB>>::make (a.x + b, a.y + b, a.z + b);
}

// Addition: vector + vector (component-wise)
template <NVec3 N3A, NVec3 N3B>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr typename Vec3BinaryOpTraits<N3A, N3B>::ReturnType operator+ (
    const N3A& a, const N3B& b)
{
    return Vec3BinaryOpTraits<N3A, N3B>::make (a.x + b.x, a.y + b.y, a.z + b.z);
}

// Subtraction: scalar - vector (broadcasts scalar to all components)
template <Number NA, NVec3 N3B>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr typename Vec3BinaryOpTraits<NVec3_t<NA>, N3B>::ReturnType operator- (
    const NA a, const N3B& b)
{
    return Vec3BinaryOpTraits<NVec3_t<NA>, N3B>::make (a - b.x, a - b.y, a - b.z);
}

// Subtraction: vector - scalar (broadcasts scalar to all components)
template <NVec3 N3A, Number NB>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr typename Vec3BinaryOpTraits<N3A, NVec3_t<NB>>::ReturnType operator- (
    const N3A& a, const NB b)
{
    return Vec3BinaryOpTraits<N3A, NVec3_t<NB>>::make (a.x - b, a.y - b, a.z - b);
}

// Subtraction: vector - vector (component-wise)
template <NVec3 N3A, NVec3 N3B>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr typename Vec3BinaryOpTraits<N3A, N3B>::ReturnType operator- (
    const N3A& a, const N3B& b)
{
    return Vec3BinaryOpTraits<N3A, N3B>::make (a.x - b.x, a.y - b.y, a.z - b.z);
}

// Multiplication: scalar * vector (broadcasts scalar to all components)
template <Number NA, NVec3 N3B>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr typename Vec3BinaryOpTraits<NVec3_t<NA>, N3B>::ReturnType operator* (
    const NA a, const N3B& b)
{
    return Vec3BinaryOpTraits<NVec3_t<NA>, N3B>::make (a * b.x, a * b.y, a * b.z);
}

// Multiplication: vector * scalar (broadcasts scalar to all components)
template <NVec3 N3A, Number NB>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr typename Vec3BinaryOpTraits<N3A, NVec3_t<NB>>::ReturnType operator* (
    const N3A& a, const NB b)
{
    return Vec3BinaryOpTraits<N3A, NVec3_t<NB>>::make (a.x * b, a.y * b, a.z * b);
}

// Multiplication: vector * vector (component-wise, Hadamard product)
template <NVec3 N3A, NVec3 N3B>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr typename Vec3BinaryOpTraits<N3A, N3B>::ReturnType operator* (
    const N3A& a, const N3B& b)
{
    return Vec3BinaryOpTraits<N3A, N3B>::make (a.x * b.x, a.y * b.y, a.z * b.z);
}

// Division: vector / scalar (optimized with reciprocal multiplication for floats)
template <NVec3 N3A, Number NB>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr typename Vec3BinaryOpTraits<N3A, NVec3_t<NB>>::ReturnType operator/ (
    const N3A& a, const NB b)
{
    if constexpr (std::is_floating_point_v<NB>)
    {
        const NB rb = static_cast<NB> (1) / b;
        return Vec3BinaryOpTraits<N3A, NVec3_t<NB>>::make (a.x * rb, a.y * rb, a.z * rb);
    }
    else
    {
        return Vec3BinaryOpTraits<N3A, NVec3_t<NB>>::make (a.x / b, a.y / b, a.z / b);
    }
}

// Division: vector / vector (component-wise)
template <NVec3 N3A, NVec3 N3B>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr typename Vec3BinaryOpTraits<N3A, N3B>::ReturnType operator/ (
    const N3A& a, const N3B& b)
{
    return Vec3BinaryOpTraits<N3A, N3B>::make (a.x / b.x, a.y / b.y, a.z / b.z);
}

// Modulo: vector % scalar (integer vectors only)
template <NVec3I N3A, std::integral NB>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr typename Vec3BinaryOpTraits<N3A, NVec3_t<NB>>::ReturnType operator% (
    const N3A& a, const NB b)
{
    return Vec3BinaryOpTraits<N3A, NVec3_t<NB>>::make (a.x % b, a.y % b, a.z % b);
}

// Modulo: vector % vector (component-wise, integer vectors only)
template <NVec3I N3A, NVec3I N3B>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr typename Vec3BinaryOpTraits<N3A, N3B>::ReturnType operator% (
    const N3A& a, const N3B& b)
{
    return Vec3BinaryOpTraits<N3A, N3B>::make (a.x % b.x, a.y % b.y, a.z % b.z);
}

// Left shift: scalar << vector (integer types only)
template <std::integral NA, NVec3I N3B>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr typename Vec3BinaryOpTraits<NVec3_t<NA>, N3B>::ReturnType operator<< (
    const NA a, const N3B& b)
{
    return Vec3BinaryOpTraits<NVec3_t<NA>, N3B>::make (a << b.x, a << b.y, a << b.z);
}

// Left shift: vector << scalar (integer types only)
template <NVec3I N3A, std::integral NB>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr typename Vec3BinaryOpTraits<N3A, NVec3_t<NB>>::ReturnType operator<< (
    const N3A& a, const NB b)
{
    return Vec3BinaryOpTraits<N3A, NVec3_t<NB>>::make (a.x << b, a.y << b, a.z << b);
}

// Left shift: vector << vector (component-wise, integer types only)
template <NVec3I N3A, NVec3I N3B>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr typename Vec3BinaryOpTraits<N3A, N3B>::ReturnType operator<< (
    const N3A& a, const N3B& b)
{
    return Vec3BinaryOpTraits<N3A, N3B>::make (a.x << b.x, a.y << b.y, a.z << b.z);
}

// Right shift: scalar >> vector (integer types only)
template <std::integral NA, NVec3I N3B>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr typename Vec3BinaryOpTraits<NVec3_t<NA>, N3B>::ReturnType operator>> (
    const NA a, const N3B& b)
{
    return Vec3BinaryOpTraits<NVec3_t<NA>, N3B>::make (a >> b.x, a >> b.y, a >> b.z);
}

// Right shift: vector >> scalar (integer types only)
template <NVec3I N3A, std::integral NB>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr typename Vec3BinaryOpTraits<N3A, NVec3_t<NB>>::ReturnType operator>> (
    const N3A& a, const NB b)
{
    return Vec3BinaryOpTraits<N3A, NVec3_t<NB>>::make (a.x >> b, a.y >> b, a.z >> b);
}

// Right shift: vector >> vector (component-wise, integer types only)
template <NVec3I N3A, NVec3I N3B>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr typename Vec3BinaryOpTraits<N3A, N3B>::ReturnType operator>> (
    const N3A& a, const N3B& b)
{
    return Vec3BinaryOpTraits<N3A, N3B>::make (a.x >> b.x, a.y >> b.y, a.z >> b.z);
}

// Compound assignment operators for 3D vectors

// Addition assignment: vector += vector
template <NVec3 N3A, NVec3 N3B>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr void operator+= (
    N3A& a, const N3B& b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}

// Subtraction assignment: vector -= vector
template <NVec3 N3A, NVec3 N3B>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr void operator-= (
    N3A& a, const N3B& b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
}

// Multiplication assignment: vector *= scalar
template <NVec3 N3A, Number NB>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr void operator*= (
    N3A& a, const NB& b)
{
    a.x *= b;
    a.y *= b;
    a.z *= b;
}

// Multiplication assignment: vector *= vector (component-wise)
template <NVec3 N3A, NVec3 N3B>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr void operator*= (
    N3A& a, const N3B& b)
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
}

// Division assignment: vector /= scalar (optimized for floating-point)
template <NVec3 N3A, Number NB>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr void operator/= (
    N3A& a, const NB& b)
{
    if constexpr (std::is_floating_point_v<NB>)
    {
        const NB rb = static_cast<NB> (1) / b;
        a.x *= rb;
        a.y *= rb;
        a.z *= rb;
    }
    else
    {
        a.x /= b;
        a.y /= b;
        a.z /= b;
    }
}

// Division assignment: vector /= vector (component-wise)
template <NVec3 N3A, NVec3 N3B>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr void operator/= (
    N3A& a, const N3B& b)
{
    a.x /= b.x;
    a.y /= b.y;
    a.z /= b.z;
}

// Modulo assignment: vector %= scalar (integer vectors only)
template <NVec3I N3A, std::integral NB>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr void operator%= (
    N3A& a, const NB& b)
{
    a.x %= b;
    a.y %= b;
    a.z %= b;
}

// Modulo assignment: vector %= vector (component-wise, integer vectors only)
template <NVec3I N3A, NVec3I N3B>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr void operator%= (
    N3A& a, const N3B& b)
{
    a.x %= b.x;
    a.y %= b.y;
    a.z %= b.z;
}

// Left shift assignment: vector <<= scalar (integer vectors only)
template <NVec3I N3A, std::integral NB>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr void operator<<= (
    N3A& a, const NB& b)
{
    a.x <<= b;
    a.y <<= b;
    a.z <<= b;
}

// Left shift assignment: vector <<= vector (component-wise, integer vectors only)
template <NVec3I N3A, NVec3I N3B>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr void operator<<= (
    N3A& a, const N3B& b)
{
    a.x <<= b.x;
    a.y <<= b.y;
    a.z <<= b.z;
}

// Right shift assignment: vector >>= scalar (integer vectors only)
template <NVec3I N3A, std::integral NB>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr void operator>>= (
    N3A& a, const NB& b)
{
    a.x >>= b;
    a.y >>= b;
    a.z >>= b;
}

// Right shift assignment: vector >>= vector (component-wise, integer vectors only)
template <NVec3I N3A, NVec3I N3B>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr void operator>>= (
    N3A& a, const N3B& b)
{
    a.x >>= b.x;
    a.y >>= b.y;
    a.z >>= b.z;
}

// Calculate Euclidean length/magnitude of 3D vector
template <NVec3 N3>
CUDA_COMMON_FUNCTION CUDA_INLINE decltype (N3::x) length (const N3& v)
{
    return std::sqrt (pow2 (v.x) + pow2 (v.y) + pow2 (v.z));
}

// Component-wise minimum: scalar and vector
template <Number NA, NVec4 N3B>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr typename Vec3BinaryOpTraits<NVec3_t<NA>, N3B>::ReturnType min (
    const NA a, const N3B& b)
{
    using UType = GetBinOpResultType_t<NA, decltype (N3B::x)>;
    return Vec3BinaryOpTraits<NVec3_t<NA>, N3B>::make (
        stc::min<UType> (a, b.x), stc::min<UType> (a, b.y), stc::min<UType> (a, b.z));
}

// Component-wise minimum: vector and scalar
template <NVec3 N3A, Number NB>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr typename Vec3BinaryOpTraits<N3A, NVec3_t<NB>>::ReturnType min (
    const N3A& a, const NB b)
{
    using UType = GetBinOpResultType_t<decltype (N3A::x), NB>;
    return Vec3BinaryOpTraits<N3A, NVec3_t<NB>>::make (
        stc::min<UType> (a.x, b), stc::min<UType> (a.y, b), stc::min<UType> (a.z, b));
}

// Component-wise minimum: vector and vector
template <NVec3 N3A, NVec3 N3B>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr typename Vec3BinaryOpTraits<N3A, N3B>::ReturnType min (
    const N3A& a, const N3B& b)
{
    using UType = GetBinOpResultType_t<decltype (N3A::x), decltype (N3B::x)>;
    return Vec3BinaryOpTraits<N3A, N3B>::make (
        stc::min<UType> (a.x, b.x), stc::min<UType> (a.y, b.y), stc::min<UType> (a.z, b.z));
}

// Component-wise maximum: scalar and vector
template <Number NA, NVec4 N3B>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr typename Vec3BinaryOpTraits<NVec3_t<NA>, N3B>::ReturnType max (
    const NA a, const N3B& b)
{
    using UType = GetBinOpResultType_t<NA, decltype (N3B::x)>;
    return Vec3BinaryOpTraits<NVec3_t<NA>, N3B>::make (
        stc::max<UType> (a, b.x), stc::max<UType> (a, b.y), stc::max<UType> (a, b.z));
}

// Component-wise maximum: vector and scalar
template <NVec3 N3A, Number NB>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr typename Vec3BinaryOpTraits<N3A, NVec3_t<NB>>::ReturnType max (
    const N3A& a, const NB b)
{
    using UType = GetBinOpResultType_t<decltype (N3A::x), NB>;
    return Vec3BinaryOpTraits<N3A, NVec3_t<NB>>::make (
        stc::max<UType> (a.x, b), stc::max<UType> (a.y, b), stc::max<UType> (a.z, b));
}

// Component-wise maximum: vector and vector
template <NVec3 N3A, NVec3 N3B>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr typename Vec3BinaryOpTraits<N3A, N3B>::ReturnType max (
    const N3A& a, const N3B& b)
{
    using UType = GetBinOpResultType_t<decltype (N3A::x), decltype (N3B::x)>;
    return Vec3BinaryOpTraits<N3A, N3B>::make (
        stc::max<UType> (a.x, b.x), stc::max<UType> (a.y, b.y), stc::max<UType> (a.z, b.z));
}

// Dot product of two 3D vectors
template <NVec3 N3A, NVec3 N3B>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr GetBinOpResultType_t<decltype (N3A::x), decltype (N3B::x)> dot (
    const N3A& a, const N3B& b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

// 4D Vector Operators

// Unary plus operator for 4D vectors (returns copy)
template <NVec4 N4>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr N4 operator+ (const N4& v)
{
    return v;
}

// Unary minus operator for 4D vectors (component-wise negation)
template <NVec4 N4>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr N4 operator- (const N4& v)
{
    return GetNVec4<decltype (N4::x)>::make (-v.x, -v.y, -v.z, -v.w);
}

// Equality comparison for 4D vectors
template <NVec4 N4A, NVec4 N4B>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr bool operator== (
    const N4A& a, const N4B& b)
{
    return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;
}

// Inequality comparison for 4D vectors
template <NVec4 N4A, NVec4 N4B>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr bool operator!= (
    const N4A& a, const N4B& b)
{
    return a.x != b.x || a.y != b.y || a.z != b.z || a.w != b.w;
}

// Addition: scalar + vector (broadcasts scalar to all components)
template <Number NA, NVec4 N4B>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr typename Vec4BinaryOpTraits<NVec4_t<NA>, N4B>::ReturnType operator+ (
    const NA a, const N4B& b)
{
    return Vec4BinaryOpTraits<NVec4_t<NA>, N4B>::make (a + b.x, a + b.y, a + b.z, a + b.w);
}

// Addition: vector + scalar (broadcasts scalar to all components)
template <NVec4 N4A, Number NB>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr typename Vec4BinaryOpTraits<N4A, NVec4_t<NB>>::ReturnType operator+ (
    const N4A& a, const NB b)
{
    return Vec4BinaryOpTraits<N4A, NVec4_t<NB>>::make (a.x + b, a.y + b, a.z + b, a.w + b);
}

// Addition: vector + vector (component-wise)
template <NVec4 N4A, NVec4 N4B>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr typename Vec4BinaryOpTraits<N4A, N4B>::ReturnType operator+ (
    const N4A& a, const N4B& b)
{
    return Vec4BinaryOpTraits<N4A, N4B>::make (a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

// Subtraction: scalar - vector (broadcasts scalar to all components)
template <Number NA, NVec4 N4B>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr typename Vec4BinaryOpTraits<NVec4_t<NA>, N4B>::ReturnType operator- (
    const NA a, const N4B& b)
{
    return Vec4BinaryOpTraits<NVec4_t<NA>, N4B>::make (a - b.x, a - b.y, a - b.z, a - b.w);
}

// Subtraction: vector - scalar (broadcasts scalar to all components)
template <NVec4 N4A, Number NB>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr typename Vec4BinaryOpTraits<N4A, NVec4_t<NB>>::ReturnType operator- (
    const N4A& a, const NB b)
{
    return Vec4BinaryOpTraits<N4A, NVec4_t<NB>>::make (a.x - b, a.y - b, a.z - b, a.w - b);
}

// Subtraction: vector - vector (component-wise)
template <NVec4 N4A, NVec4 N4B>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr typename Vec4BinaryOpTraits<N4A, N4B>::ReturnType operator- (
    const N4A& a, const N4B& b)
{
    return Vec4BinaryOpTraits<N4A, N4B>::make (a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}

// Multiplication: scalar * vector (broadcasts scalar to all components)
template <Number NA, NVec4 N4B>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr typename Vec4BinaryOpTraits<NVec4_t<NA>, N4B>::ReturnType operator* (
    const NA a, const N4B& b)
{
    return Vec4BinaryOpTraits<NVec4_t<NA>, N4B>::make (a * b.x, a * b.y, a * b.z, a * b.w);
}

// Multiplication: vector * scalar (broadcasts scalar to all components)
template <NVec4 N4A, Number NB>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr typename Vec4BinaryOpTraits<N4A, NVec4_t<NB>>::ReturnType operator* (
    const N4A& a, const NB b)
{
    return Vec4BinaryOpTraits<N4A, NVec4_t<NB>>::make (a.x * b, a.y * b, a.z * b, a.w * b);
}

// Multiplication: vector * vector (component-wise, Hadamard product)
template <NVec4 N4A, NVec4 N4B>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr typename Vec4BinaryOpTraits<N4A, N4B>::ReturnType operator* (
    const N4A& a, const N4B& b)
{
    return Vec4BinaryOpTraits<N4A, N4B>::make (a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}

// Division: vector / scalar (optimized with reciprocal multiplication for floats)
template <NVec4 N4A, Number NB>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr typename Vec4BinaryOpTraits<N4A, NVec4_t<NB>>::ReturnType operator/ (
    const N4A& a, const NB b)
{
    if constexpr (std::is_floating_point_v<NB>)
    {
        const NB rb = static_cast<NB> (1) / b;
        return Vec4BinaryOpTraits<N4A, NVec4_t<NB>>::make (a.x * rb, a.y * rb, a.z * rb, a.w * rb);
    }
    else
    {
        return Vec4BinaryOpTraits<N4A, NVec4_t<NB>>::make (a.x / b, a.y / b, a.z / b, a.w / b);
    }
}

// Division: vector / vector (component-wise)
template <NVec4 N4A, NVec4 N4B>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr typename Vec4BinaryOpTraits<N4A, N4B>::ReturnType operator/ (
    const N4A& a, const N4B& b)
{
    return Vec4BinaryOpTraits<N4A, N4B>::make (a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
}

// Modulo: vector % scalar (integer vectors only)
template <NVec4I N4A, std::integral NB>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr typename Vec4BinaryOpTraits<N4A, NVec4_t<NB>>::ReturnType operator% (
    const N4A& a, const NB b)
{
    return Vec4BinaryOpTraits<N4A, NVec4_t<NB>>::make (a.x % b, a.y % b, a.z % b, a.w % b);
}

// Modulo: vector % vector (component-wise, integer vectors only)
template <NVec4I N4A, NVec4I N4B>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr typename Vec4BinaryOpTraits<N4A, N4B>::ReturnType operator% (
    const N4A& a, const N4B& b)
{
    return Vec4BinaryOpTraits<N4A, N4B>::make (a.x % b.x, a.y % b.y, a.z % b.z, a.w % b.w);
}

// Left shift: scalar << vector (integer types only)
template <std::integral NA, NVec4I N4B>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr typename Vec4BinaryOpTraits<NVec4_t<NA>, N4B>::ReturnType operator<< (
    const NA a, const N4B& b)
{
    return Vec4BinaryOpTraits<NVec4_t<NA>, N4B>::make (a << b.x, a << b.y, a << b.z, a << b.w);
}

// Left shift: vector << scalar (integer types only)
template <NVec4I N4A, std::integral NB>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr typename Vec4BinaryOpTraits<N4A, NVec4_t<NB>>::ReturnType operator<< (
    const N4A& a, const NB b)
{
    return Vec4BinaryOpTraits<N4A, NVec4_t<NB>>::make (a.x << b, a.y << b, a.z << b, a.w << b);
}

// Left shift: vector << vector (component-wise, integer types only)
template <NVec4I N4A, NVec4I N4B>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr typename Vec4BinaryOpTraits<N4A, N4B>::ReturnType operator<< (
    const N4A& a, const N4B& b)
{
    return Vec4BinaryOpTraits<N4A, N4B>::make (a.x << b.x, a.y << b.y, a.z << b.z, a.w << b.w);
}

// Right shift: scalar >> vector (integer types only)
template <std::integral NA, NVec4I N4B>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr typename Vec4BinaryOpTraits<NVec4_t<NA>, N4B>::ReturnType operator>> (
    const NA a, const N4B& b)
{
    return Vec4BinaryOpTraits<NVec4_t<NA>, N4B>::make (a >> b.x, a >> b.y, a >> b.z, a >> b.w);
}

// Right shift: vector >> scalar (integer types only)
template <NVec4I N4A, std::integral NB>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr typename Vec4BinaryOpTraits<N4A, NVec4_t<NB>>::ReturnType operator>> (
    const N4A& a, const NB b)
{
    return Vec4BinaryOpTraits<N4A, NVec4_t<NB>>::make (a.x >> b, a.y >> b, a.z >> b, a.w >> b);
}

// Right shift: vector >> vector (component-wise, integer types only)
template <NVec4I N4A, NVec4I N4B>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr typename Vec4BinaryOpTraits<N4A, N4B>::ReturnType operator>> (
    const N4A& a, const N4B& b)
{
    return Vec4BinaryOpTraits<N4A, N4B>::make (a.x >> b.x, a.y >> b.y, a.z >> b.z, a.w >> b.w);
}

// Compound assignment operators for 4D vectors

// Addition assignment: vector += vector
template <NVec4 N4A, NVec4 N4B>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr void operator+= (
    N4A& a, const N4B& b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
}

// Subtraction assignment: vector -= vector
template <NVec4 N4A, NVec4 N4B>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr void operator-= (
    N4A& a, const N4B& b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    a.w -= b.w;
}

// Multiplication assignment: vector *= scalar
template <NVec4 N4A, Number NB>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr void operator*= (
    N4A& a, const NB& b)
{
    a.x *= b;
    a.y *= b;
    a.z *= b;
    a.w *= b;
}

// Multiplication assignment: vector *= vector (component-wise)
template <NVec4 N4A, NVec4 N4B>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr void operator*= (
    N4A& a, const N4B& b)
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
    a.w *= b.w;
}

// Division assignment: vector /= scalar (optimized for floating-point)
template <NVec4 N4A, Number NB>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr void operator/= (
    N4A& a, const NB& b)
{
    if constexpr (std::is_floating_point_v<NB>)
    {
        const NB rb = static_cast<NB> (1) / b;
        a.x *= rb;
        a.y *= rb;
        a.z *= rb;
        a.w *= rb;
    }
    else
    {
        a.x /= b;
        a.y /= b;
        a.z /= b;
        a.w /= b;
    }
}

// Division assignment: vector /= vector (component-wise)
template <NVec4 N4A, NVec4 N4B>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr void operator/= (
    N4A& a, const N4B& b)
{
    a.x /= b.x;
    a.y /= b.y;
    a.z /= b.z;
    a.w /= b.w;
}

// Modulo assignment: vector %= scalar (integer vectors only)
template <NVec4I N4A, std::integral NB>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr void operator%= (
    N4A& a, const NB& b)
{
    a.x %= b;
    a.y %= b;
    a.z %= b;
    a.w %= b;
}

// Modulo assignment: vector %= vector (component-wise, integer vectors only)
template <NVec4I N4A, NVec4I N4B>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr void operator%= (
    N4A& a, const N4B& b)
{
    a.x %= b.x;
    a.y %= b.y;
    a.z %= b.z;
    a.w %= b.w;
}

// Left shift assignment: vector <<= scalar (integer vectors only)
template <NVec4I N4A, std::integral NB>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr void operator<<= (
    N4A& a, const NB& b)
{
    a.x <<= b;
    a.y <<= b;
    a.z <<= b;
    a.w <<= b;
}

// Left shift assignment: vector <<= vector (component-wise, integer vectors only)
template <NVec4I N4A, NVec4I N4B>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr void operator<<= (
    N4A& a, const N4B& b)
{
    a.x <<= b.x;
    a.y <<= b.y;
    a.z <<= b.z;
    a.w <<= b.w;
}

// Right shift assignment: vector >>= scalar (integer vectors only)
template <NVec4I N4A, std::integral NB>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr void operator>>= (
    N4A& a, const NB& b)
{
    a.x >>= b;
    a.y >>= b;
    a.z >>= b;
    a.w >>= b;
}

// Right shift assignment: vector >>= vector (component-wise, integer vectors only)
template <NVec4I N4A, NVec4I N4B>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr void operator>>= (
    N4A& a, const N4B& b)
{
    a.x >>= b.x;
    a.y >>= b.y;
    a.z >>= b.z;
    a.w >>= b.w;
}

// Extract xyz components from 4D vector as 3D vector
template <NVec4 N4>
CUDA_COMMON_FUNCTION CUDA_INLINE NVec3_t<decltype (N4::x)> getXYZ (const N4& v)
{
    return GetNVec3<decltype (N4::x)>::make (v.x, v.y, v.z);
}

// Component-wise minimum: scalar and vector
template <Number NA, NVec4 N4B>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr typename Vec4BinaryOpTraits<NVec4_t<NA>, N4B>::ReturnType min (
    const NA a, const N4B& b)
{
    using UType = GetBinOpResultType_t<NA, decltype (N4B::x)>;
    return Vec4BinaryOpTraits<NVec4_t<NA>, N4B>::make (
        stc::min<UType> (a, b.x), stc::min<UType> (a, b.y), stc::min<UType> (a, b.z), stc::min<UType> (a, b.w));
}

// Component-wise minimum: vector and scalar
template <NVec4 N4A, Number NB>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr typename Vec4BinaryOpTraits<N4A, NVec4_t<NB>>::ReturnType min (
    const N4A& a, const NB b)
{
    using UType = GetBinOpResultType_t<decltype (N4A::x), NB>;
    return Vec4BinaryOpTraits<N4A, NVec4_t<NB>>::make (
        stc::min<UType> (a.x, b), stc::min<UType> (a.y, b), stc::min<UType> (a.z, b), stc::min<UType> (a.w, b));
}

// Component-wise minimum: vector and vector
template <NVec4 N4A, NVec4 N4B>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr typename Vec4BinaryOpTraits<N4A, N4B>::ReturnType min (
    const N4A& a, const N4B& b)
{
    using UType = GetBinOpResultType_t<decltype (N4A::x), decltype (N4B::x)>;
    return Vec4BinaryOpTraits<N4A, N4B>::make (
        stc::min<UType> (a.x, b.x), stc::min<UType> (a.y, b.y), stc::min<UType> (a.z, b.z), stc::min<UType> (a.w, b.w));
}

// Component-wise maximum: scalar and vector
template <Number NA, NVec4 N4B>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr typename Vec4BinaryOpTraits<NVec4_t<NA>, N4B>::ReturnType max (
    const NA a, const N4B& b)
{
    using UType = GetBinOpResultType_t<NA, decltype (N4B::x)>;
    return Vec4BinaryOpTraits<NVec4_t<NA>, N4B>::make (
        stc::max<UType> (a, b.x), stc::max<UType> (a, b.y), stc::max<UType> (a, b.z), stc::max<UType> (a, b.w));
}

// Component-wise maximum: vector and scalar
template <NVec4 N4A, Number NB>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr typename Vec4BinaryOpTraits<N4A, NVec4_t<NB>>::ReturnType max (
    const N4A& a, const NB b)
{
    using UType = GetBinOpResultType_t<decltype (N4A::x), NB>;
    return Vec4BinaryOpTraits<N4A, NVec4_t<NB>>::make (
        stc::max<UType> (a.x, b), stc::max<UType> (a.y, b), stc::max<UType> (a.z, b), stc::max<UType> (a.w, b));
}

// Component-wise maximum: vector and vector
template <NVec4 N4A, NVec4 N4B>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr typename Vec4BinaryOpTraits<N4A, N4B>::ReturnType max (
    const N4A& a, const N4B& b)
{
    using UType = GetBinOpResultType_t<decltype (N4A::x), decltype (N4B::x)>;
    return Vec4BinaryOpTraits<N4A, N4B>::make (
        stc::max<UType> (a.x, b.x), stc::max<UType> (a.y, b.y), stc::max<UType> (a.z, b.z), stc::max<UType> (a.w, b.w));
}

// Dot product of two 4D vectors
template <NVec4 N4A, NVec4 N4B>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr GetBinOpResultType_t<decltype (N4A::x), decltype (N4B::x)> dot (
    const N4A& a, const N4B& b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

