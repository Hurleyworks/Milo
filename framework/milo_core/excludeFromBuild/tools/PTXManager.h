#pragma once

#include "../../milo_core.h"

// Class that manages loading, selection, and access to PTX/OptiXIR data
// Handles architecture-specific variants and maintains a non-static lifecycle
// tied to the RenderContext to ensure proper reinitialization after scene clears
class PTXManager
{
 public:
    // Constructor initializes with render context
    PTXManager();

    // Destructor ensures proper cleanup
    ~PTXManager();

    // Initialize the manager with resource paths
    void initialize (const std::filesystem::path& resourceFolder);

    // Reset state for reinitialization
    void reset();

    // Get PTX/OptiXIR data for a kernel name
    // Returns a vector containing the PTX/OptiXIR data
    // If useEmbedded is true, attempts to load from embedded data
    // If useEmbedded is false, loads from file
    std::vector<char> getPTXData (const std::string& kernelName, bool useEmbedded = true);

    // Check if a kernel is available in either embedded data or file system
    bool isKernelAvailable (const std::string& kernelName);

    // Get the compute capability of the current GPU as a single integer
    // For example, SM 8.6 (RTX 3090) returns 86
    int getComputeCapability() const;

    // Get the format (ptx or optixir) for a given kernel
    std::string getFormat (const std::string& kernelName);

    // Get the list of available kernels
    std::vector<std::string> getAvailableKernels();

 private:
    // Helper method to load PTX data from embedded resources
    std::vector<char> loadEmbeddedPTX (const std::string& kernelName);

    // Helper method to load PTX data from filesystem
    std::vector<char> loadFilePTX (const std::string& kernelName);

    // Reference to the render context
    // RenderContextPtr ctx = nullptr;

    // Path to resources directory
    std::filesystem::path resourcePath;

    // Initialization state flag (not static)
    bool initialized = false;

    // Cache of detected compute capability
    int computeCapability = 0;

    // Build mode (Debug/Release)
    std::string buildMode;

    // Cache of available kernels to avoid repeated checks
    std::vector<std::string> availableKernelCache;
};