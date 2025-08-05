#pragma once

#include "../../engine_core.h"
#include "../../generated/embedded_ptx.h" // Generated header

// Simple wrapper around the generated embedded_ptx.h functionality
// This class is primarily maintained for backward compatibility
// New code should use PTXManager directly
class EmbeddedPTX
{
 public:
    // Get PTX/OPTIXIR data by name without extension
    static std::pair<const unsigned char*, size_t> getData (const std::string& name)
    {
#ifdef NDEBUG
        try
        {
            // Release mode - try to use embedded data
            return getEmbeddedData (name);
        }
        catch (const std::exception& e)
        {
            LOG (WARNING) << "Failed to get embedded PTX data: " << e.what();
            throw;
        }
#else
        // Development mode - return null pair to indicate fallback to file loading
        return {nullptr, 0};
#endif
    }

    // Get the format (ptx or optixir) for a given kernel
    static std::string getFormat (const std::string& name, int computeCapability)
    {
#ifdef EMBEDDED_PTX_AVAILABLE
        return embedded::getFormatForCompute (name, computeCapability);
#else
        return "unknown";
#endif
    }

    // Check if a kernel is available
    static bool hasKernel (const std::string& name)
    {
#ifdef EMBEDDED_PTX_AVAILABLE
        return embedded::hasKernel (name);
#else
        return false;
#endif
    }

    // Initialize embedded data - no static state
    static void initialize()
    {
#ifdef EMBEDDED_PTX_AVAILABLE
        embedded::initialize();
#endif
    }

 private:
    static std::pair<const unsigned char*, size_t> getEmbeddedData (const std::string& name)
    {
#ifndef EMBEDDED_PTX_AVAILABLE
        throw std::runtime_error ("Embedded PTX data not available");
#else
        int computeCapability = 0;

        // Try to detect compute capability
        CUdevice device;
        if (cuDeviceGet (&device, 0) == CUDA_SUCCESS)
        {
            int major = 0, minor = 0;
            cuDeviceGetAttribute (&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
            cuDeviceGetAttribute (&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device);
            computeCapability = major * 10 + minor;
        }

        // Fall back to standard compute capability if detection failed
        if (computeCapability == 0)
        {
            computeCapability = 86; // Default to RTX 3090
        }

        // Let the embedded namespace handle the selection
        return embedded::getBestMatchForCompute (name, computeCapability);
#endif
    }
};