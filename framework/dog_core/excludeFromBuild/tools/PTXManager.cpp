#include "PTXManager.h"
#include "EmbeddedPTX.h"
#include "../../../engine_core/generated/embedded_ptx.h"

// Constructor initializes with render context
PTXManager::PTXManager() :
    initialized (false)
{
    LOG (DBUG) << "Creating PTXManager";

    // Determine build mode at construction time
#ifdef NDEBUG
    buildMode = "Release";
#else
    buildMode = "Debug";
#endif
}

PTXManager::~PTXManager()
{
    LOG (DBUG) << "Destroying PTXManager";

    try
    {
        // Reset state, clearing any cached data
        reset();

        // Explicitly clear resource paths
        resourcePath = std::filesystem::path();
    }
    catch (const std::exception& e)
    {
        LOG (WARNING) << "Error during PTXManager destruction: " << e.what();
    }
}

// Initialize the manager with resource paths
void PTXManager::initialize (const std::filesystem::path& resourceFolder)
{
    LOG (DBUG) << "Initializing PTXManager with resource folder: " << resourceFolder.string();

    // Store resource path
    resourcePath = resourceFolder;

    // Cache the compute capability for this GPU
    // computeCapability = ctx->getComputeCapabilityAsInt();
    LOG (DBUG) << "Detected compute capability: " << computeCapability;

    // Initialize embedded PTX namespace only if in Release mode
#ifdef EMBEDDED_PTX_AVAILABLE
    try
    {
        // Call embedded initialize (doesn't rely on static state)
        embedded::initialize();
        LOG (DBUG) << "Embedded PTX data initialized";
    }
    catch (const std::exception& e)
    {
        LOG (WARNING) << "Failed to initialize embedded PTX data: " << e.what();
    }
#else
    LOG (DBUG) << "Embedded PTX data not available in this build";
#endif

    // Pre-cache available kernels
    availableKernelCache = getAvailableKernels();
    LOG (DBUG) << "Found " << availableKernelCache.size() << " available kernels";

    initialized = true;
}

// Reset state for reinitialization
void PTXManager::reset()
{
    LOG (DBUG) << "Resetting PTXManager";

    try
    {
        // Clear cached data
        availableKernelCache.clear();

        // Reset initialization state
        initialized = false;

        // Reset compute capability cache
        computeCapability = 0;
    }
    catch (const std::exception& e)
    {
        LOG (WARNING) << "Error during PTXManager reset: " << e.what();
    }
}
// Get PTX/OptiXIR data for a kernel name
std::vector<char> PTXManager::getPTXData (const std::string& kernelName, bool useEmbedded)
{
    LOG (DBUG) << "Getting PTX data for kernel: " << kernelName
               << " (useEmbedded=" << (useEmbedded ? "true" : "false") << ")";

    if (!initialized)
    {
        throw std::runtime_error ("PTXManager not initialized");
    }

    if (useEmbedded)
    {
        // Try embedded first
        try
        {
            return loadEmbeddedPTX (kernelName);
        }
        catch (const std::exception& e)
        {
            LOG (WARNING) << "Failed to load embedded PTX for " << kernelName
                          << ": " << e.what() << ", falling back to file";
            useEmbedded = false;
        }
    }

    if (!useEmbedded)
    {
        // Fall back to file loading
        return loadFilePTX (kernelName);
    }

    // This should not be reached, but just in case
    throw std::runtime_error ("Failed to load PTX data for " + kernelName);
}

// Helper method to load PTX data from embedded resources
std::vector<char> PTXManager::loadEmbeddedPTX (const std::string& kernelName)
{
#ifdef EMBEDDED_PTX_AVAILABLE
    // Check if kernel is available
    if (!embedded::hasKernel (kernelName))
    {
        throw std::runtime_error ("Kernel not available in embedded PTX: " + kernelName);
    }

    // Get best matching data for current compute capability
    LOG (DBUG) << "Getting best match for " << kernelName
               << " at compute capability " << computeCapability;

    auto [data_ptr, data_size] = embedded::getBestMatchForCompute (kernelName, computeCapability);

    if (!data_ptr || data_size == 0)
    {
        throw std::runtime_error ("Failed to get embedded data for " + kernelName);
    }

    LOG (DBUG) << "Successfully loaded embedded data for " << kernelName
               << " (" << data_size << " bytes)";

    // Convert to vector
    return std::vector<char> (data_ptr, data_ptr + data_size);
#else
    // Embedded PTX not available
    throw std::runtime_error ("Embedded PTX data not available in this build");
#endif
}

// Helper method to load PTX data from filesystem
std::vector<char> PTXManager::loadFilePTX (const std::string& kernelName)
{
    // Construct path based on build mode
    std::filesystem::path ptxPath = resourcePath / "ptx" / buildMode;

    // For RTX 3090 (sm_86), look in sm_86 folder first
    if (computeCapability == 86)
    {
        std::filesystem::path specificPath = ptxPath / "sm_86" / (kernelName + ".optixir");
        if (std::filesystem::exists (specificPath))
        {
            LOG (DBUG) << "Using RTX 3090 specific path: " << specificPath.string();
            return readBinaryFile (specificPath);
        }

        specificPath = ptxPath / "sm_86" / (kernelName + ".ptx");
        if (std::filesystem::exists (specificPath))
        {
            LOG (DBUG) << "Using RTX 3090 specific path: " << specificPath.string();
            return readBinaryFile (specificPath);
        }
    }

    // Try .optixir first
    std::filesystem::path optixirPath = ptxPath / (kernelName + ".optixir");
    if (std::filesystem::exists (optixirPath))
    {
        LOG (DBUG) << "Loading OptiXIR from: " << optixirPath.string();
        return readBinaryFile (optixirPath);
    }

    // Fall back to .ptx
    std::filesystem::path ptxFilePath = ptxPath / (kernelName + ".ptx");
    if (std::filesystem::exists (ptxFilePath))
    {
        LOG (DBUG) << "Loading PTX from: " << ptxFilePath.string();
        return readBinaryFile (ptxFilePath);
    }

    throw std::runtime_error ("PTX file not found for " + kernelName);
}

// Check if a kernel is available in either embedded data or file system
bool PTXManager::isKernelAvailable (const std::string& kernelName)
{
    if (!initialized)
    {
        LOG (WARNING) << "PTXManager not initialized when checking for kernel: " << kernelName;
        return false;
    }

    // Check embedded availability
#ifdef EMBEDDED_PTX_AVAILABLE
    if (embedded::hasKernel (kernelName))
    {
        return true;
    }
#endif

    // Check file availability
    std::filesystem::path optixirPath = resourcePath / "ptx" / buildMode / (kernelName + ".optixir");
    std::filesystem::path ptxPath = resourcePath / "ptx" / buildMode / (kernelName + ".ptx");

    return std::filesystem::exists (optixirPath) || std::filesystem::exists (ptxPath);
}

// Get the compute capability of the current GPU
int PTXManager::getComputeCapability() const
{
    return computeCapability;
}

// Get the format (ptx or optixir) for a given kernel
std::string PTXManager::getFormat (const std::string& kernelName)
{
    if (!initialized)
    {
        throw std::runtime_error ("PTXManager not initialized");
    }

#ifdef EMBEDDED_PTX_AVAILABLE
    try
    {
        // Get format from embedded data
        return embedded::getFormatForCompute (kernelName, computeCapability);
    }
    catch (const std::exception&)
    {
        // Fall through to file-based check
    }
#endif

    // Check file extensions
    std::filesystem::path optixirPath = resourcePath / "ptx" / buildMode / (kernelName + ".optixir");
    if (std::filesystem::exists (optixirPath))
    {
        return "optixir";
    }

    std::filesystem::path ptxPath = resourcePath / "ptx" / buildMode / (kernelName + ".ptx");
    if (std::filesystem::exists (ptxPath))
    {
        return "ptx";
    }

    return "unknown";
}

// Get the list of available kernels
std::vector<std::string> PTXManager::getAvailableKernels()
{
    if (!initialized && !resourcePath.empty())
    {
        LOG (WARNING) << "PTXManager not initialized when getting available kernels";
    }

    std::vector<std::string> result;
    std::set<std::string> uniqueKernels;

    // First check embedded kernels
#ifdef EMBEDDED_PTX_AVAILABLE
    try
    {
        // Known core kernels that should be available
        const std::vector<std::string> knownKernels = {
            "optix_kernels",
            "optix_pick_kernels",
            "copy_buffers",
            "deform"};

        for (const auto& kernel : knownKernels)
        {
            if (embedded::hasKernel (kernel))
            {
                uniqueKernels.insert (kernel);
            }
        }
    }
    catch (const std::exception& e)
    {
        LOG (WARNING) << "Error checking embedded kernels: " << e.what();
    }
#endif

    // Then check file system
    if (!resourcePath.empty())
    {
        std::filesystem::path ptxDir = resourcePath / "ptx" / buildMode;

        if (std::filesystem::exists (ptxDir) && std::filesystem::is_directory (ptxDir))
        {
            for (const auto& entry : std::filesystem::directory_iterator (ptxDir))
            {
                if (entry.is_regular_file())
                {
                    std::string filename = entry.path().filename().string();
                    std::string extension = entry.path().extension().string();

                    if (extension == ".ptx" || extension == ".optixir")
                    {
                        // Remove extension to get kernel name
                        std::string kernelName = filename.substr (0, filename.length() - extension.length());
                        uniqueKernels.insert (kernelName);
                    }
                }
            }
        }
    }

    // Convert set to vector
    result.assign (uniqueKernels.begin(), uniqueKernels.end());
    return result;
}