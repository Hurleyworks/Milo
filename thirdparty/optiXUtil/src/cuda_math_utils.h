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

// CUDA-compatible mathematical utilities and standard library extensions
//
// This header provides a comprehensive collection of mathematical functions
// and utilities optimized for both CPU and GPU execution contexts. The design
// enables seamless code sharing between host and device code while leveraging
// platform-specific optimizations.
//
// Key Design Principles:
// - Dual-target compatibility: Functions work identically on CPU and GPU
// - Platform-specific optimizations: Uses CUDA intrinsics when available
// - Template-based design: Generic implementations for different numeric types
// - Constexpr support: Compile-time evaluation where possible
// - Performance focus: Optimized implementations for high-throughput scenarios
//
// Library Organization:
//
// stc Namespace (std-complementary):
// - Drop-in replacements for std library functions unavailable in CUDA
// - Conditional implementation based on compilation target
// - Floating-point classification and bit manipulation utilities
//
// Basic Mathematical Utilities:
// - Optimized power functions (pow2, pow3, pow4, pow5)
// - Linear interpolation (lerp) for generic types
// - Compile-time constant expressions where possible
//
// Integer Arithmetic Utilities:
// - Memory alignment calculations for GPU memory management
// - Floor division and modulus operations with proper negative handling
// - Essential for buffer management and indexing operations
//
// Bit Manipulation Functions:
// - Leading/trailing zero count with CUDA intrinsic optimization
// - Population count for efficient bit operations
// - Power-of-2 calculations for memory allocation and indexing
// - Nth set bit finding for sparse data structures
//
// Float-Integer Conversion Utilities:
// - Ordered comparison support through integer representation
// - Essential for atomic operations on floating-point values
// - Maintains IEEE 754 comparison semantics in integer domain
//
// Common Applications:
// - High-performance GPU computing and graphics programming
// - Memory management and buffer alignment
// - Bit manipulation for compact data structures
// - Atomic operations on floating-point data
// - Mathematical computations in ray tracing and physics simulations
// - Spatial data structures and geometric algorithms
//
// All functions are designed for optimal performance in parallel computing
// environments with careful attention to divergence minimization and
// memory access patterns.

// std-complementary functions for CUDA
namespace stc
{
    // Swap two values with platform-optimized implementation
    template <typename T>
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr void swap (T& a, T& b)
    {
#if defined(__CUDA_ARCH__)
        T temp = a;
        a = b;
        b = temp;
#else
        std::swap (a, b);
#endif
    }

    // Return the smaller of two values
    template <typename T>
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr T min (const T& a, const T& b)
    {
        return a < b ? a : b;
    }

    // Return the larger of two values
    template <typename T>
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr T max (const T& a, const T& b)
    {
        return a > b ? a : b;
    }

    // Clamp value to specified range [_min, _max]
    template <typename T>
    CUDA_COMMON_FUNCTION CUDA_INLINE constexpr T clamp (const T& x, const T& _min, const T& _max)
    {
        return min (max (x, _min), _max);
    }

    // Test if floating-point value is infinite
    template <std::floating_point F>
    CUDA_COMMON_FUNCTION CUDA_INLINE bool isinf (const F x)
    {
#if defined(__CUDA_ARCH__)
        return static_cast<bool> (::isinf (x));
#else
        return std::isinf (x);
#endif
    }

    // Test if floating-point value is NaN (Not a Number)
    template <std::floating_point F>
    CUDA_COMMON_FUNCTION CUDA_INLINE bool isnan (const F x)
    {
#if defined(__CUDA_ARCH__)
        return static_cast<bool> (::isnan (x));
#else
        return std::isnan (x);
#endif
    }

    // Test if floating-point value is finite (not infinite or NaN)
    template <std::floating_point F>
    CUDA_COMMON_FUNCTION CUDA_INLINE bool isfinite (const F x)
    {
#if defined(__CUDA_ARCH__)
        return static_cast<bool> (::isfinite (x));
#else
        return std::isfinite (x);
#endif
    }

    // Compute sine and cosine simultaneously (more efficient than separate calls)
    template <std::floating_point F>
    CUDA_COMMON_FUNCTION CUDA_INLINE void sincos (const F x, F* const s, F* const c)
    {
#if defined(__CUDA_ARCH__)
        ::sincosf (x, s, c);
#else
        *s = std::sin (x);
        *c = std::cos (x);
#endif
    }

    // Reinterpret bit pattern of source type as destination type
    template <typename DstType, typename SrcType>
    CUDA_COMMON_FUNCTION CUDA_INLINE DstType bit_cast (const SrcType& x)
    {
#if defined(__CUDA_ARCH__)
        if constexpr (std::is_same_v<SrcType, int32_t> && std::is_same_v<DstType, float>)
            return __int_as_float (x);
        else if constexpr (std::is_same_v<SrcType, uint32_t> && std::is_same_v<DstType, float>)
            return __uint_as_float (x);
        else if constexpr (std::is_same_v<SrcType, float> && std::is_same_v<DstType, int32_t>)
            return __float_as_int (x);
        else if constexpr (std::is_same_v<SrcType, float> && std::is_same_v<DstType, uint32_t>)
            return __float_as_uint (x);
        static_assert (sizeof (DstType) == sizeof (SrcType), "Sizes do not match.");
        union
        {
            SrcType s;
            DstType d;
        } alias;
        alias.s = x;
        return alias.d;
#else
        return std::bit_cast<DstType> (x);
#endif
    }
} // namespace stc

// Compute x squared (x²)
template <typename T>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr T pow2 (const T& x)
{
    return x * x;
}

// Compute x cubed (x³)
template <typename T>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr T pow3 (const T& x)
{
    return x * pow2 (x);
}

// Compute x to the fourth power (x?)
template <typename T>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr T pow4 (const T& x)
{
    return pow2 (pow2 (x));
}

// Compute x to the fifth power (x?)
template <typename T>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr T pow5 (const T& x)
{
    return x * pow4 (x);
}

// Linear interpolation between two values
template <typename T, std::floating_point F>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr T lerp (const T& v0, const T& v1, const F t)
{
    return (1 - t) * v0 + t * v1;
}

// Align integer value up to next multiple of specified alignment
template <std::integral IntType>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr IntType alignUp (
    const IntType value, const uint32_t alignment)
{
    return static_cast<IntType> ((value + alignment - 1) / alignment * alignment);
}

// Floor division with proper handling of negative numbers
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr int32_t floorDiv (
    const int32_t value, const uint32_t modulus)
{
    return (value < 0 ? (value - static_cast<int32_t> (modulus - 1)) : value) / static_cast<int32_t> (modulus);
}

// Floor modulus with proper handling of negative numbers
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr uint32_t floorMod (
    const int32_t value, const uint32_t modulus)
{
    int32_t r = value % static_cast<int32_t> (modulus);
    return r < 0 ? r + modulus : r;
}

// Count trailing zeros in 32-bit integer
CUDA_COMMON_FUNCTION CUDA_INLINE uint32_t tzcnt (const uint32_t x)
{
#if defined(__CUDA_ARCH__)
    return __clz (__brev (x));
#else
    return _tzcnt_u32 (x);
#endif
}

// Count trailing zeros in 32-bit integer (constexpr version)
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr uint32_t tzcntConst (const uint32_t x)
{
    for (uint32_t i = 0; i < 32; ++i)
    {
        if ((x >> i) & 0b1)
            return i;
    }
    return 32;
}

// Count leading zeros in 32-bit integer
CUDA_COMMON_FUNCTION CUDA_INLINE uint32_t lzcnt (const uint32_t x)
{
#if defined(__CUDA_ARCH__)
    return __clz (x);
#else
    return _lzcnt_u32 (x);
#endif
}

// Count leading zeros in 32-bit integer (constexpr version)
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr uint32_t lzcntConst (const uint32_t x)
{
    for (uint32_t i = 0; i < 32; ++i)
    {
        if ((x >> (31 - i)) & 0b1)
            return i;
    }
    return 32;
}

// Count number of set bits in 32-bit integer (population count)
CUDA_COMMON_FUNCTION CUDA_INLINE int32_t popcnt (const uint32_t x)
{
#if defined(__CUDA_ARCH__)
    return __popc (x);
#else
    return _mm_popcnt_u32 (x);
#endif
}

// Find exponent of largest power of 2 less than or equal to x
CUDA_COMMON_FUNCTION CUDA_INLINE uint32_t prevPowOf2Exponent (const uint32_t x)
{
    if (x == 0)
        return 0;
    return 31 - lzcnt (x);
}

// Find exponent of smallest power of 2 greater than or equal to x
CUDA_COMMON_FUNCTION CUDA_INLINE uint32_t nextPowOf2Exponent (const uint32_t x)
{
    if (x == 0)
        return 0;
    return 32 - lzcnt (x - 1);
}

// Find largest power of 2 less than or equal to x
CUDA_COMMON_FUNCTION CUDA_INLINE uint32_t prevPowerOf2 (const uint32_t x)
{
    if (x == 0)
        return 0;
    return 1 << prevPowOf2Exponent (x);
}

// Find smallest power of 2 greater than or equal to x
CUDA_COMMON_FUNCTION CUDA_INLINE uint32_t nextPowerOf2 (const uint32_t x)
{
    if (x == 0)
        return 0;
    return 1 << nextPowOf2Exponent (x);
}

// Round up to next multiple of power of 2 (specified by exponent)
template <std::integral IntType>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr IntType nextMultiplesForPowOf2 (
    const IntType x, const uint32_t exponent)
{
    const IntType mask = (1 << exponent) - 1;
    return (x + mask) & ~mask;
}

// Calculate how many times power of 2 fits into rounded-up value
template <std::integral IntType>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr IntType nextMultiplierForPowOf2 (
    const IntType x, const uint32_t exponent)
{
    return nextMultiplesForPowOf2 (x, exponent) >> exponent;
}

// Find the position of the nth set bit in a 32-bit integer
CUDA_COMMON_FUNCTION CUDA_INLINE uint32_t nthSetBit (uint32_t value, int32_t n)
{
    uint32_t idx = 0;
    int32_t count;
    if (n >= popcnt (value))
        return 0xFFFFFFFF;

    for (uint32_t width = 16; width >= 1; width >>= 1)
    {
        if (value == 0)
            return 0xFFFFFFFF;

        const uint32_t mask = (1 << width) - 1;
        count = popcnt (value & mask);
        if (n >= count)
        {
            value >>= width;
            n -= count;
            idx += width;
        }
    }

    return idx;
}

// Convert float to integer preserving IEEE 754 comparison order
CUDA_COMMON_FUNCTION CUDA_INLINE int32_t floatToOrderedInt (const float fVal)
{
#if defined(__CUDA_ARCH__)
    const int32_t iVal = __float_as_int (fVal);
#else
    const int32_t iVal = std::bit_cast<int32_t> (fVal);
#endif
    return (iVal >= 0) ? iVal : iVal ^ 0x7FFF'FFFF;
}

// Convert ordered integer back to float (inverse of floatToOrderedInt)
CUDA_COMMON_FUNCTION CUDA_INLINE float orderedIntToFloat (const int32_t iVal)
{
    const int32_t orgBits = (iVal >= 0) ? iVal : iVal ^ 0x7FFF'FFFF;
#if defined(__CUDA_ARCH__)
    return __int_as_float (orgBits);
#else
    return std::bit_cast<float> (orgBits);
#endif
}

// Convert float to unsigned integer preserving IEEE 754 comparison order
CUDA_COMMON_FUNCTION CUDA_INLINE uint32_t floatToOrderedUInt (const float fVal)
{
#if defined(__CUDA_ARCH__)
    const uint32_t uiVal = __float_as_uint (fVal);
#else
    const uint32_t uiVal = std::bit_cast<uint32_t> (fVal);
#endif
    return uiVal ^ (uiVal < 0x8000'0000 ? 0x8000'0000 : 0xFFFF'FFFF);
}

// Convert ordered unsigned integer back to float (inverse of floatToOrderedUInt)
CUDA_COMMON_FUNCTION CUDA_INLINE float orderedUIntToFloat (const uint32_t uiVal)
{
    const uint32_t orgBits = uiVal ^ (uiVal >= 0x8000'0000 ? 0x8000'0000 : 0xFFFF'FFFF);
#if defined(__CUDA_ARCH__)
    return __uint_as_float (orgBits);
#else
    return std::bit_cast<float> (orgBits);
#endif
}