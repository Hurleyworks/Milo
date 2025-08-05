#pragma once

// Platform defines
#if defined(_WIN32) || defined(_WIN64)
#define HP_Platform_Windows
#if defined(_MSC_VER)
#define HP_Platform_Windows_MSVC
#endif
#elif defined(__APPLE__)
#define HP_Platform_macOS
#endif

#ifdef _DEBUG
#define ENABLE_ASSERT
#define DEBUG_SELECT(A, B) A
#else
#define DEBUG_SELECT(A, B) B
#endif

#if defined(HP_Platform_Windows_MSVC)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#define WIN32_LEAN_AND_MEAN // Excludes rarely-used Windows headers
#define NOCOMM              // Excludes serial communication API
#define NODRIVERS           // Excludes driver APIs
#define NOSERVICE           // Excludes Service Controller APIs
#define NOSOUND             // Excludes sound APIs
#define NOKANJI             // Excludes Kanji support
#define NOMCX               // Excludes Modem Configuration APIs
#define NOCRYPT             // Excludes cryptography APIs
#include <Windows.h>
#undef near
#undef far
#undef RGB
#endif

// #includes
#if defined(__CUDA_ARCH__)
#else
#include <cstdio>
#include <cstdlib>
#include <immintrin.h>
#endif

#include <cstdint>
#include <cmath>
#include <algorithm>
#include <type_traits>
#include <bit>

#include "optixu_on_cudau.h"

#ifdef HP_Platform_Windows_MSVC
#if defined(__CUDA_ARCH__)
#define devPrintf(fmt, ...) printf (fmt, ##__VA_ARGS__);
#else
void devPrintf (const char* fmt, ...);
#endif
#else
#define devPrintf(fmt, ...) printf (fmt, ##__VA_ARGS__);
#endif

#if defined(__CUDA_ARCH__)
#define __Assert(expr, fmt, ...) \
    do \
    { \
        if (!(expr)) \
        { \
            devPrintf ("%s @%s: %u:\n", #expr, __FILE__, __LINE__); \
            devPrintf (fmt "\n", ##__VA_ARGS__); \
            assert (false); \
        } \
    } while (0)
#else
#define __Assert(expr, fmt, ...) \
    do \
    { \
        if (!(expr)) \
        { \
            devPrintf ("%s @%s: %u:\n", #expr, __FILE__, __LINE__); \
            devPrintf (fmt "\n", ##__VA_ARGS__); \
            abort(); \
        } \
    } while (0)
#endif

#ifdef ENABLE_ASSERT
#define Assert(expr, fmt, ...) __Assert (expr, fmt, ##__VA_ARGS__)
#else
#define Assert(expr, fmt, ...)
#endif

#define Assert_Release(expr, fmt, ...) __Assert (expr, fmt, ##__VA_ARGS__)

#define Assert_ShouldNotBeCalled() __Assert (false, "Should not be called!")
#define Assert_NotImplemented() __Assert (false, "Not implemented yet!")

#define V2FMT "%g, %g"
#define V3FMT "%g, %g, %g"
#define V4FMT "%g, %g, %g, %g"
#define v2print(v) (v).x, (v).y
#define v3print(v) (v).x, (v).y, (v).z
#define v4print(v) (v).x, (v).y, (v).z, (v).w
#define rgbprint(v) (v).r, (v).g, (v).b


inline void devPrintf (const char* fmt, ...)
{
    va_list args;
    va_start (args, fmt);
    char str[4096];
    vsnprintf_s (str, sizeof (str), _TRUNCATE, fmt, args);
    va_end (args);
    OutputDebugString (str);
}

template <typename T>
concept Number = std::integral<T> || std::floating_point<T>;

template <typename T>
concept Number32bit =
    std::is_same_v<T, int32_t> ||
    std::is_same_v<T, uint32_t> ||
    std::is_same_v<T, float>;

#if !defined(PURE_CUDA) || defined(CUDAU_CODE_COMPLETION)
CUDA_DEVICE_FUNCTION CUDA_INLINE bool isCursorPixel();
CUDA_DEVICE_FUNCTION CUDA_INLINE bool getDebugPrintEnabled();
#endif

template <typename T, size_t size>
CUDA_COMMON_FUNCTION CUDA_INLINE constexpr size_t lengthof (const T (&array)[size])
{
    return size;
}