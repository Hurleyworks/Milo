#pragma once

// GPUManager handles detection, information gathering, and selection of CUDA-capable GPUs.
// It enumerates all available devices, filters those that meet OptiX requirements,
// and manages the currently selected GPU. This class provides the necessary abstraction
// for multi-GPU support in the RiPR plugin, allowing users to choose the optimal
// device for rendering operations.

#include "../../engine_core.h"

// Forward declarations
class PropertyService;

class GPUManager
{
 public:
    // Information structure for CUDA devices
    struct GPUInfo
    {
        int deviceIndex;               // CUDA device index
        std::string name;              // Device name
        size_t totalMemory;            // Total memory in bytes
        size_t freeMemory;             // Free memory in bytes
        size_t usedMemory;             // Used memory in bytes
        int computeCapabilityMajor;    // CUDA compute capability major version
        int computeCapabilityMinor;    // CUDA compute capability minor version
        int driverVersion;             // CUDA driver version
        bool meetsMinimumRequirements; // Whether device meets OptiX requirements
        bool isActive;                 // Whether this is the currently active device
    };

    // Constructor with property service reference for settings persistence
    explicit GPUManager (const PropertyService& properties);
    ~GPUManager();

    // Initialize CUDA and enumerate available GPUs
    bool initialize();

    // Refresh GPU information (e.g., to update memory stats)
    void refreshGPUStats();

    // Get information about all GPUs
    const std::vector<GPUInfo>& getGPUInfo() const;

    // Get currently selected GPU index
    int getSelectedGPUIndex() const;

    // Set selected GPU index
    bool setSelectedGPUIndex (int index);

    // Get number of available GPUs
    int getGPUCount() const;

    // Check if selected GPU is valid for OptiX
    bool isSelectedGPUValid() const;

    // Get formatted description string for a GPU
    std::string getGPUDescription (int index) const;

    // Check if changing GPU requires renderer restart
    bool requiresRendererRestart() const;

 private:
    std::vector<GPUInfo> gpuInfo;      // Information about all detected GPUs
    int selectedGPUIndex = 0;          // Currently selected GPU index
    const PropertyService& properties; // Reference to property service for settings
    bool initialized = false;          // Whether CUDA has been initialized
    bool cudaAvailable = false;        // Whether CUDA is available on this system
    bool useFakeGPUs = false;

    // Initialize CUDA driver
    bool initializeCUDA();

    // Check if GPU meets OptiX requirements
    bool checkGPURequirements (int deviceIndex, GPUInfo& info);

    // Update properties with current selection
    void updateProperties();

    // Get current active GPU memory stats
    void updateActiveGPUMemoryStats();

    // For testing: Generate fake GPU entries
    void generateFakeGPUs (int count);
};