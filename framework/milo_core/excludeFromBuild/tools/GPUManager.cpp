#include "GPUManager.h"
#include "../common/common_host.h" // For CUDADRV_CHECK macro and other utilities

GPUManager::GPUManager (const PropertyService& properties) :
    properties (properties),
    selectedGPUIndex (0),
    initialized (false),
    cudaAvailable (false)
{
    // Try to load the selected GPU index from properties
    try
    {
        selectedGPUIndex = properties.renderProps->getVal<int> (RenderKey::SelectedGPUIndex);

        // Check if fake GPUs should be used (for testing)
        useFakeGPUs = properties.renderProps->getVal<bool> (RenderKey::UseFakeGPUs);

        // Validate the index is in range
        if (selectedGPUIndex < 0 || selectedGPUIndex >= static_cast<int> (gpuInfo.size()))
        {
            LOG (WARNING) << "Invalid GPU index in properties: " << selectedGPUIndex
                          << ", defaulting to 0";
            selectedGPUIndex = 0;
        }
    }
    catch (...)
    {
        // If property doesn't exist or there's an error, use default
        selectedGPUIndex = 0;
    }
}

#if 0
GPUManager::GPUManager (const PropertyService& properties) :
    properties (properties),
    selectedGPUIndex (0),
    initialized (false),
    cudaAvailable (false)
{
    // Try to load the selected GPU index from properties
    try
    {
        selectedGPUIndex = properties.renderProps->getVal<int> (RenderKey::SelectedGPUIndex);

        // Validate the index is in range
        if (selectedGPUIndex < 0 || selectedGPUIndex >= static_cast<int> (gpuInfo.size()))
        {
            LOG (WARNING) << "Invalid GPU index in properties: " << selectedGPUIndex
                          << ", defaulting to 0";
            selectedGPUIndex = 0;
        }
    }
    catch (...)
    {
        // If property doesn't exist or there's an error, use default
        selectedGPUIndex = 0;
    }
}
#endif

GPUManager::~GPUManager()
{
    // No need to explicitly clean up CUDA here as the RenderContext
    // handles CUDA context cleanup
}

bool GPUManager::initialize()
{
    if (initialized)
    {
        return true;
    }

    // Always read the current setting when initializing
    try
    {
        useFakeGPUs = properties.renderProps->getVal<bool> (RenderKey::UseFakeGPUs);
        LOG (DBUG) << "Initializing GPU manager with useFakeGPUs = " << (useFakeGPUs ? "true" : "false");
    }
    catch (...)
    {
        useFakeGPUs = DEFAULT_USE_FAKE_GPUS;
        LOG (DBUG) << "Failed to read UseFakeGPUs property, using default: " << (useFakeGPUs ? "true" : "false");
    }

    // Initialize CUDA
    cudaAvailable = initializeCUDA();
    if (!cudaAvailable)
    {
        LOG (WARNING) << "CUDA initialization failed, no GPUs will be available";
        return false;
    }

    // Clear previous information
    gpuInfo.clear();

    // Get number of devices
    int deviceCount = 0;
    try
    {
        CUDADRV_CHECK (cuDeviceGetCount (&deviceCount));
    }
    catch (const std::exception& e)
    {
        LOG (WARNING) << "Failed to get CUDA device count: " << e.what();
        return false;
    }

    if (deviceCount == 0)
    {
        LOG (WARNING) << "No CUDA devices found";
        return false;
    }

    LOG (DBUG) << "Found " << deviceCount << " CUDA device(s)";

    // Enumerate each device
    for (int i = 0; i < deviceCount; i++)
    {
        GPUInfo info;
        info.deviceIndex = i;

        // Get device handle
        CUdevice device;
        try
        {
            CUDADRV_CHECK (cuDeviceGet (&device, i));
        }
        catch (const std::exception& e)
        {
            LOG (WARNING) << "Failed to get handle for device " << i << ": " << e.what();
            continue;
        }

        // Get device name
        char deviceName[256];
        try
        {
            CUDADRV_CHECK (cuDeviceGetName (deviceName, sizeof (deviceName), device));
            info.name = deviceName;
        }
        catch (const std::exception& e)
        {
            LOG (WARNING) << "Failed to get name for device " << i << ": " << e.what();
            info.name = "Unknown CUDA Device";
        }

        // Get compute capability
        try
        {
            CUDADRV_CHECK (cuDeviceGetAttribute (&info.computeCapabilityMajor,
                                                 CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                                                 device));
            CUDADRV_CHECK (cuDeviceGetAttribute (&info.computeCapabilityMinor,
                                                 CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                                                 device));
        }
        catch (const std::exception& e)
        {
            LOG (WARNING) << "Failed to get compute capability for device " << i << ": " << e.what();
            info.computeCapabilityMajor = 0;
            info.computeCapabilityMinor = 0;
        }

        // Get total memory
        try
        {
            CUDADRV_CHECK (cuDeviceTotalMem (&info.totalMemory, device));
            // We don't have free memory info yet - will be updated when device is active
            info.freeMemory = 0;
        }
        catch (const std::exception& e)
        {
            LOG (WARNING) << "Failed to get memory info for device " << i << ": " << e.what();
            info.totalMemory = 0;
            info.freeMemory = 0;
        }

        // Get driver version
        try
        {
            CUDADRV_CHECK (cuDriverGetVersion (&info.driverVersion));
        }
        catch (const std::exception& e)
        {
            LOG (WARNING) << "Failed to get driver version: " << e.what();
            info.driverVersion = 0;
        }

        // Check if device meets OptiX requirements
        info.meetsMinimumRequirements = checkGPURequirements (i, info);

        // Is this the currently active device?
        info.isActive = false; // Will be set when a device is selected

        // Add to our list
        gpuInfo.push_back (info);

        LOG (DBUG) << "GPU " << i << ": " << info.name
                   << ", Compute: " << info.computeCapabilityMajor << "." << info.computeCapabilityMinor
                   << ", Memory: " << (info.totalMemory / (1024 * 1024)) << " MB"
                   << ", Driver: " << (info.driverVersion / 1000) << "." << (info.driverVersion % 100)
                   << ", Compatible: " << (info.meetsMinimumRequirements ? "Yes" : "No");
    }

    // Try to load the selected GPU index from properties
    try
    {
        selectedGPUIndex = properties.renderProps->getVal<int> (RenderKey::SelectedGPUIndex);

        // Validate the index is in range
        if (selectedGPUIndex < 0 || selectedGPUIndex >= static_cast<int> (gpuInfo.size()))
        {
            LOG (WARNING) << "Invalid GPU index in properties: " << selectedGPUIndex
                          << ", defaulting to 0";
            selectedGPUIndex = 0;
            updateProperties();
        }
    }
    catch (...)
    {
        // If property doesn't exist or there's an error, use default
        selectedGPUIndex = 0;
        updateProperties();
    }

    // Update properties with current GPU count
    try
    {
        properties.renderProps->setValue (RenderKey::GPUCount, static_cast<int> (gpuInfo.size()));
    }
    catch (...)
    {
        LOG (WARNING) << "Failed to set GPU count property";
    }

    // Add fake GPUs for testing if enabled
    if (useFakeGPUs)
    {
        int fakeCount = properties.renderProps->getVal<int> (RenderKey::FakeGPUCount);
        if (fakeCount > 0)
        {
            generateFakeGPUs (fakeCount);

            // Update properties with new GPU count including fake ones
            properties.renderProps->setValue (RenderKey::GPUCount, static_cast<int> (gpuInfo.size()));

            LOG (DBUG) << "Added " << fakeCount << " fake GPUs for testing. Total GPUs: " << gpuInfo.size();
        }
    }

    // Mark the selected GPU as active
    if (!gpuInfo.empty())
    {
        gpuInfo[selectedGPUIndex].isActive = true;
    }

    initialized = true;
    return true;
    initialized = true;
    return true;
}

void GPUManager::refreshGPUStats()
{
    if (!initialized || gpuInfo.empty())
    {
        initialize();
        return;
    }

    // Update memory information for the active GPU
    updateActiveGPUMemoryStats();

    // Log updated memory stats
    if (selectedGPUIndex >= 0 && selectedGPUIndex < static_cast<int> (gpuInfo.size()))
    {
        const auto& info = gpuInfo[selectedGPUIndex];
        LOG (DBUG) << "GPU " << selectedGPUIndex << " (" << info.name << ") Memory: "
                   << (info.usedMemory / (1024 * 1024)) << " MB used, "
                   << (info.freeMemory / (1024 * 1024)) << " MB free of "
                   << (info.totalMemory / (1024 * 1024)) << " MB total";
    }
}

const std::vector<GPUManager::GPUInfo>& GPUManager::getGPUInfo() const
{
    return gpuInfo;
}

int GPUManager::getSelectedGPUIndex() const
{
    return selectedGPUIndex;
}
bool GPUManager::setSelectedGPUIndex (int index)
{
    if (!initialized)
    {
        return false;
    }

    if (index < 0 || index >= static_cast<int> (gpuInfo.size()))
    {
        LOG (WARNING) << "Invalid GPU index: " << index;
        return false;
    }

    // Check if this is a fake GPU (index >= 1000)
    bool isFakeGPU = (gpuInfo[index].deviceIndex >= 1000);

    if (isFakeGPU)
    {
        LOG (WARNING) << "Attempted to select fake GPU: " << gpuInfo[index].name;

        if (!useFakeGPUs)
        {
            // Only allow selecting fake GPUs when fake GPU mode is enabled
            LOG (WARNING) << "Selecting fake GPUs not allowed in production. Ignoring selection.";
            return false;
        }
        else
        {
            LOG (WARNING) << "Fake GPU selected for testing purposes only.";
            // Update UI state only - mark previously selected as inactive
            if (selectedGPUIndex >= 0 && selectedGPUIndex < static_cast<int> (gpuInfo.size()))
            {
                gpuInfo[selectedGPUIndex].isActive = false;
            }

            // Mark new selection as active for UI
            selectedGPUIndex = index;
            gpuInfo[selectedGPUIndex].isActive = true;

            // DON'T update properties from backend thread!

            // Return false to indicate no renderer reinitialization needed
            return false;
        }
    }

    // Normal real GPU selection logic - only update if the selection has changed
    if (selectedGPUIndex != index)
    {
        // Mark previous selection as inactive
        if (selectedGPUIndex >= 0 && selectedGPUIndex < static_cast<int> (gpuInfo.size()))
        {
            gpuInfo[selectedGPUIndex].isActive = false;
        }

        // Update selection
        selectedGPUIndex = index;

        // Mark new selection as active
        gpuInfo[selectedGPUIndex].isActive = true;

        // DON'T update properties from backend thread!

        return true; // Signal that renderer needs to be reinitialized
    }

    return false; // No change needed
}

#if 0

bool GPUManager::setSelectedGPUIndex (int index)
{
    if (!initialized)
    {
        return false;
    }

    if (gpuInfo[index].deviceIndex >= 1000)
    {
        selectedGPUIndex = 0;
        return false;
    }
       

    if (index < 0 || index >= static_cast<int> (gpuInfo.size()))
    {
        LOG (WARNING) << "Invalid GPU index: " << index;
        return false;
    }

    // Check if this is a fake GPU (index >= 1000)
    if (gpuInfo[index].deviceIndex >= 1000)
    {
        LOG (WARNING) << "Attempted to select fake GPU: " << gpuInfo[index].name;

        if (!useFakeGPUs)
        {
            // Only allow selecting fake GPUs when fake GPU mode is enabled
            LOG (WARNING) << "Selecting fake GPUs not allowed in production. Ignoring selection.";
            return false;
        }
        else
        {
            LOG (WARNING) << "Fake GPU selected for testing purposes only.";
            // Continue with selection for testing purposes
        }
    }
    else if (!gpuInfo[index].meetsMinimumRequirements)
    {
        LOG (WARNING) << "Selected GPU does not meet minimum requirements";
        return false;
    }

    // Only update if the selection has changed
    if (selectedGPUIndex != index)
    {
        // Mark previous selection as inactive
        if (selectedGPUIndex >= 0 && selectedGPUIndex < static_cast<int> (gpuInfo.size()))
        {
            gpuInfo[selectedGPUIndex].isActive = false;
        }

        // Update selection
        selectedGPUIndex = index;

        // Mark new selection as active
        gpuInfo[selectedGPUIndex].isActive = true;

        // Update properties
        updateProperties();
    }

    return true;
}


bool GPUManager::setSelectedGPUIndex (int index)
{
    if (!initialized)
    {
        return false;
    }

    if (index < 0 || index >= static_cast<int> (gpuInfo.size()))
    {
        LOG (WARNING) << "Invalid GPU index: " << index;
        return false;
    }

    // Check if this is a fake GPU (index ? 1000)
    if (gpuInfo[index].deviceIndex >= 1000)
    {
        LOG (WARNING) << "Attempted to select fake GPU: " << gpuInfo[index].name;

        if (!useFakeGPUs)
        {
            // Only allow selecting fake GPUs when fake GPU mode is enabled
            LOG (WARNING) << "Selecting fake GPUs not allowed in production. Ignoring selection.";
            return false;
        }
        else
        {
            LOG (WARNING) << "Fake GPU selected for testing purposes only.";
            // Continue with selection for testing purposes
        }
    }

    if (!gpuInfo[index].meetsMinimumRequirements)
    {
        LOG (WARNING) << "Selected GPU does not meet minimum requirements";
        return false;
    }

    // Only update if the selection has changed
    if (selectedGPUIndex != index)
    {
        // Mark previous selection as inactive
        if (selectedGPUIndex >= 0 && selectedGPUIndex < static_cast<int> (gpuInfo.size()))
        {
            gpuInfo[selectedGPUIndex].isActive = false;
        }

        // Update selection
        selectedGPUIndex = index;

        // Mark new selection as active
        gpuInfo[selectedGPUIndex].isActive = true;

        // Update properties
        updateProperties();
    }

    return true;
}

bool GPUManager::setSelectedGPUIndex (int index)
{
    if (!initialized)
    {
        return false;
    }

    if (index < 0 || index >= static_cast<int> (gpuInfo.size()))
    {
        LOG (WARNING) << "Invalid GPU index: " << index;
        return false;
    }

    if (!gpuInfo[index].meetsMinimumRequirements)
    {
        LOG (WARNING) << "Selected GPU does not meet minimum requirements";
        return false;
    }

    // Only update if the selection has changed
    if (selectedGPUIndex != index)
    {
        // Mark previous selection as inactive
        if (selectedGPUIndex >= 0 && selectedGPUIndex < static_cast<int> (gpuInfo.size()))
        {
            gpuInfo[selectedGPUIndex].isActive = false;
        }

        // Update selection
        selectedGPUIndex = index;

        // Mark new selection as active
        gpuInfo[selectedGPUIndex].isActive = true;

        // Update properties
        updateProperties();
    }

    return true;
}

#endif

void GPUManager::generateFakeGPUs (int count)
{
    LOG (DBUG) << "Generating " << count << " fake GPUs for testing";

    // Define a list of realistic GPU names (without the FAKE prefix)
    const std::vector<std::string> gpuModels = {
        "RTX 4090", "RTX 4080", "RTX 3090 Ti", "RTX 3080",
        "RTX A6000", "RTX A5000", "RTX A4000",
        "Tesla V100", "Tesla A100", "RTX 6000 Ada"};

    // Define realistic memory sizes (in MB)
    const std::vector<size_t> memorySizes = {
        24576, // 24 GB
        16384, // 16 GB
        12288, // 12 GB
        10240, // 10 GB
        8192,  // 8 GB
        32768, // 32 GB
        48128  // 48 GB
    };

    // Define realistic compute capabilities
    const std::vector<std::pair<int, int>> computeCapabilities = {
        {8, 9}, // SM 8.9 for newest GPUs
        {8, 6}, // SM 8.6 for RTX 30xx/40xx
        {7, 5}, // SM 7.5 for RTX 20xx
        {6, 1}, // SM 6.1 for GTX 10xx
        {9, 0}  // SM 9.0 future-looking
    };

    // Create fake GPU entries with varied specs
    for (int i = 0; i < count; i++)
    {
        GPUInfo fakeGpu;

        // Use real device index + 1000 for fake GPUs to avoid conflicts
        fakeGpu.deviceIndex = 1000 + i;

        // Select a random GPU model
        size_t modelIndex = i % gpuModels.size();
        fakeGpu.name = "[FAKE] " + gpuModels[modelIndex];

        // Select varied memory sizes
        size_t memIndex = (i + 1) % memorySizes.size();
        fakeGpu.totalMemory = memorySizes[memIndex] * 1024 * 1024; // Convert MB to bytes

        // Set a realistic free memory amount (70-90% of total)
        double freePercent = 0.7 + ((i % 3) * 0.1); // 0.7, 0.8, or 0.9
        fakeGpu.freeMemory = static_cast<size_t> (fakeGpu.totalMemory * freePercent);
        fakeGpu.usedMemory = fakeGpu.totalMemory - fakeGpu.freeMemory;

        // Set compute capability
        size_t ccIndex = i % computeCapabilities.size();
        fakeGpu.computeCapabilityMajor = computeCapabilities[ccIndex].first;
        fakeGpu.computeCapabilityMinor = computeCapabilities[ccIndex].second;

        // Set driver version (vary slightly)
        fakeGpu.driverVersion = 12000 + (i * 100); // Something like 12.0, 12.1, etc.

        // Mark as meeting requirements but not active
        fakeGpu.meetsMinimumRequirements = true;
        fakeGpu.isActive = false;

        // Add to the GPU info vector
        gpuInfo.push_back (fakeGpu);

        LOG (DBUG) << "Added fake GPU: " << fakeGpu.name
                   << ", Memory: " << (fakeGpu.totalMemory / (1024 * 1024)) << " MB"
                   << ", Compute: " << fakeGpu.computeCapabilityMajor << "." << fakeGpu.computeCapabilityMinor;
    }
}

int GPUManager::getGPUCount() const
{
    return static_cast<int> (gpuInfo.size());
}

bool GPUManager::isSelectedGPUValid() const
{
    if (!initialized || gpuInfo.empty())
    {
        return false;
    }

    if (selectedGPUIndex < 0 || selectedGPUIndex >= static_cast<int> (gpuInfo.size()))
    {
        return false;
    }

    return gpuInfo[selectedGPUIndex].meetsMinimumRequirements;
}

std::string GPUManager::getGPUDescription (int index) const
{
    if (index < 0 || index >= static_cast<int> (gpuInfo.size()))
    {
        return "Invalid GPU index";
    }

    const GPUInfo& info = gpuInfo[index];

    // Format memory in GB with 2 decimal places for better readability
    float memoryGB = static_cast<float> (info.totalMemory) / (1024.0f * 1024.0f * 1024.0f);

    return std::format ("{}: {} ({}.{}) - {:.2f} GB{}",
                        info.deviceIndex,
                        info.name,
                        info.computeCapabilityMajor,
                        info.computeCapabilityMinor,
                        memoryGB,
                        info.isActive ? " [ACTIVE]" : "");
}

bool GPUManager::requiresRendererRestart() const
{
    // Currently, any GPU change requires renderer restart
    return true;
}

bool GPUManager::initializeCUDA()
{
    try
    {
        CUDADRV_CHECK (cuInit (0));
        return true;
    }
    catch (const std::exception& e)
    {
        LOG (WARNING) << "CUDA initialization failed: " << e.what();
        return false;
    }
}

bool GPUManager::checkGPURequirements (int deviceIndex, GPUInfo& info)
{
    // Check minimum requirements for OptiX:
    // - Compute capability 6.0 or higher
    // - CUDA driver version 10.1 or higher

    if (info.computeCapabilityMajor < 6)
    {
        LOG (WARNING) << "GPU " << deviceIndex << " compute capability "
                      << info.computeCapabilityMajor << "." << info.computeCapabilityMinor
                      << " is below minimum required 6.0";
        return false;
    }

    if (info.driverVersion < 10010)
    {
        LOG (WARNING) << "CUDA driver version " << info.driverVersion
                      << " is below minimum required 10.1";
        return false;
    }

    return true;
}

void GPUManager::updateProperties()
{
    try
    {
        properties.renderProps->setValue (RenderKey::SelectedGPUIndex, selectedGPUIndex);
    }
    catch (...)
    {
        LOG (WARNING) << "Failed to update SelectedGPUIndex property";
    }
}
void GPUManager::updateActiveGPUMemoryStats()
{
    if (!initialized || gpuInfo.empty() || selectedGPUIndex < 0 ||
        selectedGPUIndex >= static_cast<int> (gpuInfo.size()))
    {
        return;
    }

    // Reference to the active GPU's info
    GPUInfo& activeInfo = gpuInfo[selectedGPUIndex];

    try
    {
        //// Use our device-specific method to get accurate stats for this device
        //GPUMemoryMonitor::UpdateMemoryStatsForDevice (selectedGPUIndex);

        //// Get the updated stats
        //GPUMemoryStats stats = GPUMemoryMonitor::getStats();

        //// Update our cached info
        //activeInfo.freeMemory = stats.freeMemory;
        //activeInfo.usedMemory = stats.usedMemory;
        //activeInfo.totalMemory = stats.totalMemory;

        //// Log success
        //LOG (DBUG) << "Updated memory stats for GPU " << selectedGPUIndex;
    }
    catch (const std::exception& e)
    {
        LOG (WARNING) << "Failed to update GPU memory stats: " << e.what();
    }
}

#if 0
void GPUManager::updateActiveGPUMemoryStats()
{
    if (!initialized || gpuInfo.empty() || selectedGPUIndex < 0 ||
        selectedGPUIndex >= static_cast<int> (gpuInfo.size()))
    {
        return;
    }

    // Get current memory information for the active GPU
    GPUInfo& activeInfo = gpuInfo[selectedGPUIndex];

    // For the active device, we can use GPUMemoryMonitor
    try
    {
        // Create a temporary context if needed
        CUcontext tempContext = nullptr;
        CUdevice device;
        bool createdTempContext = false;

        CUDADRV_CHECK (cuDeviceGet (&device, selectedGPUIndex));

        // Check if a context exists for this device
        CUcontext currentContext;
        CUdevice currentDevice;

        cuCtxGetCurrent (&currentContext);
        if (currentContext == nullptr)
        {
            // No current context, create a temporary one
            CUDADRV_CHECK (cuCtxCreate (&tempContext, 0, device));
            createdTempContext = true;
        }
        else
        {
            // Context exists, check if it's for our device
            cuCtxGetDevice (&currentDevice);
            if (currentDevice != device)
            {
                // Current context is for a different device, create a new one
                CUDADRV_CHECK (cuCtxCreate (&tempContext, 0, device));
                createdTempContext = true;
            }
        }

        // Use GPUMemoryMonitor to get stats for current context
        GPUMemoryStats stats = GPUMemoryMonitor::getStats();

        activeInfo.freeMemory = stats.freeMemory;
        activeInfo.usedMemory = stats.usedMemory;
        activeInfo.totalMemory = stats.totalMemory;

        // Destroy the temporary context if we created one
        if (createdTempContext && tempContext != nullptr)
        {
            CUDADRV_CHECK (cuCtxDestroy (tempContext));
        }
    }
    catch (const std::exception& e)
    {
        LOG (WARNING) << "Failed to update GPU memory stats: " << e.what();
    }
}
#endif