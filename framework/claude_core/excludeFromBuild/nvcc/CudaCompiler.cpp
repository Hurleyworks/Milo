#include "CudaCompiler.h"
#include <reproc++/run.hpp>

// Print compilation results
void CudaCompiler::CompilationResult::print() const
{
    LOG (INFO) << "===== CUDA Compilation Summary =====";
    LOG (INFO) << "  Compiled: " << compiled << " files";
    LOG (INFO) << "  Skipped:  " << skipped << " files (up-to-date)";
    if (failed > 0)
    {
        LOG (WARNING) << "  Failed: " << failed << " files";
        for (const auto& file : failedFiles)
        {
            LOG (WARNING) << "    - " << file;
        }
    }
    LOG (INFO) << "====================================";
}

bool CudaCompiler::needsCompilation (const std::filesystem::path& cuPath,
                                     const std::filesystem::path& ptxPath)
{
    // Force compile overrides everything
    if (forceCompile_)
    {
        LOG (DBUG) << "Force compile enabled for: " << cuPath.filename();
        return true;
    }

    // PTX doesn't exist - must compile
    if (!std::filesystem::exists (ptxPath))
    {
        LOG (INFO) << "Output not found, will compile: " << cuPath.filename();
        return true;
    }

    // Check if CU is newer than PTX
    auto cuTime = std::filesystem::last_write_time (cuPath);
    auto ptxTime = std::filesystem::last_write_time (ptxPath);

    if (cuTime > ptxTime)
    {
        LOG (INFO) << "Source newer than output, will recompile: " << cuPath.filename();
        return true;
    }

    LOG (DBUG) << "Skipping up-to-date: " << cuPath.filename();
    return false;
}

std::vector<std::filesystem::path> CudaCompiler::getCudaFilesForEngine (
    const std::filesystem::path& baseFolder,
    const std::string& engineFilter)
{
    std::vector<std::filesystem::path> result;

    if (engineFilter == "all")
    {
        // Get all .cu files from all engine folders and common

#if 0
        // Milo engine files
        std::filesystem::path miloPath = baseFolder / "engines" / "milo" / "cuda";
        if (std::filesystem::exists(miloPath)) {
            for (const auto& entry : std::filesystem::directory_iterator(miloPath)) {
                if (entry.path().extension() == ".cu") {
                    result.push_back(entry.path());
                }
            }
        }
        
        // Claudia engine files
        std::filesystem::path claudiaPath = baseFolder / "engines" / "claudia" / "cuda";
        if (std::filesystem::exists(claudiaPath)) {
            for (const auto& entry : std::filesystem::directory_iterator(claudiaPath)) {
                if (entry.path().extension() == ".cu") {
                    result.push_back(entry.path());
                }
            }
        }
#endif
        // RiPR engine files
        std::filesystem::path riprPath = baseFolder / "engines" / "ripr" / "cuda";
        if (std::filesystem::exists (riprPath))
        {
            for (const auto& entry : std::filesystem::directory_iterator (riprPath))
            {
                if (entry.path().extension() == ".cu")
                {
                    result.push_back (entry.path());
                }
            }
        }

        // Shocker engine files
        std::filesystem::path shockerPath = baseFolder / "engines" / "shocker" / "cuda";
        if (std::filesystem::exists (shockerPath))
        {
            for (const auto& entry : std::filesystem::directory_iterator (shockerPath))
            {
                if (entry.path().extension() == ".cu")
                {
                    result.push_back (entry.path());
                }
            }
        }

        // Common GPU kernels
        std::filesystem::path commonPath = baseFolder / "common" / "gpu_kernels";
        if (std::filesystem::exists (commonPath))
        {
            for (const auto& entry : std::filesystem::directory_iterator (commonPath))
            {
                if (entry.path().extension() == ".cu")
                {
                    result.push_back (entry.path());
                }
            }
        }

        // Legacy cuda folder (for backward compatibility)
        std::filesystem::path legacyPath = baseFolder / "cuda";
        if (std::filesystem::exists (legacyPath))
        {
            for (const auto& entry : std::filesystem::directory_iterator (legacyPath))
            {
                if (entry.path().extension() == ".cu")
                {
                    result.push_back (entry.path());
                }
            }
        }
    }
    else if (engineFilter == "milo")
    {
        // Only Milo engine files
        std::filesystem::path miloPath = baseFolder / "engines" / "milo" / "cuda";
        if (std::filesystem::exists (miloPath))
        {
            for (const auto& entry : std::filesystem::directory_iterator (miloPath))
            {
                if (entry.path().extension() == ".cu")
                {
                    result.push_back (entry.path());
                }
            }
        }

        // Add common files that Milo uses
        std::filesystem::path commonPath = baseFolder / "common" / "gpu_kernels";
        if (std::filesystem::exists (commonPath))
        {
            // Add compute_light_probs.cu if it exists (used by Milo)
            auto lightProbs = commonPath / "compute_light_probs.cu";
            if (std::filesystem::exists (lightProbs))
            {
                result.push_back (lightProbs);
            }
        }
    }
    else if (engineFilter == "claudia")
    {
        // Only Claudia engine files
        std::filesystem::path claudiaPath = baseFolder / "engines" / "claudia" / "cuda";
        if (std::filesystem::exists (claudiaPath))
        {
            for (const auto& entry : std::filesystem::directory_iterator (claudiaPath))
            {
                if (entry.path().extension() == ".cu")
                {
                    result.push_back (entry.path());
                }
            }
        }

        // Add common files that Claudia uses
        std::filesystem::path commonPath = baseFolder / "common" / "gpu_kernels";
        if (std::filesystem::exists (commonPath))
        {
            // Add compute_light_probs.cu if it exists (used by Claudia)
            auto lightProbs = commonPath / "compute_light_probs.cu";
            if (std::filesystem::exists (lightProbs))
            {
                result.push_back (lightProbs);
            }
        }
    }
    else if (engineFilter == "ripr")
    {
        // Only RiPR engine files
        std::filesystem::path riprPath = baseFolder / "engines" / "ripr" / "cuda";
        if (std::filesystem::exists (riprPath))
        {
            for (const auto& entry : std::filesystem::directory_iterator (riprPath))
            {
                if (entry.path().extension() == ".cu")
                {
                    result.push_back (entry.path());
                }
            }
        }

        // Add common files that RiPR uses
        std::filesystem::path commonPath = baseFolder / "common" / "gpu_kernels";
        if (std::filesystem::exists (commonPath))
        {
            // Add compute_light_probs.cu if it exists (used by RiPR)
            auto lightProbs = commonPath / "compute_light_probs.cu";
            if (std::filesystem::exists (lightProbs))
            {
                result.push_back (lightProbs);
            }
        }
    }
    else if (engineFilter == "shocker")
    {
        // Only Shocker engine files
        std::filesystem::path shockerPath = baseFolder / "engines" / "shocker" / "cuda";
        if (std::filesystem::exists (shockerPath))
        {
            for (const auto& entry : std::filesystem::directory_iterator (shockerPath))
            {
                if (entry.path().extension() == ".cu")
                {
                    result.push_back (entry.path());
                }
            }
        }

        // Add common files that Shocker uses
        std::filesystem::path commonPath = baseFolder / "common" / "gpu_kernels";
        if (std::filesystem::exists (commonPath))
        {
            // Add compute_light_probs.cu if it exists (used by Shocker)
            auto lightProbs = commonPath / "compute_light_probs.cu";
            if (std::filesystem::exists (lightProbs))
            {
                result.push_back (lightProbs);
            }
        }
    }

    LOG (DBUG) << "Found " << result.size() << " CUDA files for engine: " << engineFilter;
    return result;
}

bool CudaCompiler::compileSingleFile (const std::filesystem::path& cuFile,
                                      const std::filesystem::path& outputFolder,
                                      const std::string& architecture,
                                      const std::string& buildMode,
                                      const std::filesystem::path& shockerUtilFolder,
                                      bool generatePTX)
{
    std::string fileName = cuFile.filename().string();

    // Determine output file path
    std::string outPath = generatePTX
                              ? (outputFolder / cuFile.stem()).string() + ".ptx"
                              : (outputFolder / cuFile.stem()).string() + ".optixir";

    // Check if compilation is needed
    if (!needsCompilation (cuFile, outPath))
    {
        return false; // Skipped
    }

    LOG (INFO) << "Compiling: " << fileName << " -> " << std::filesystem::path (outPath).filename();

    // Path to nvcc exe
    std::string exe = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/bin/nvcc.exe";

    // Build nvcc arguments
    std::vector<std::string> args;
    args.push_back (exe);
    args.push_back (cuFile.string());

    if (generatePTX)
        args.push_back ("--ptx");
    else
        args.push_back ("--optix-ir");

    args.push_back ("--extended-lambda");
    args.push_back ("--use_fast_math");
    args.push_back ("--cudart");
    args.push_back ("shared");
    args.push_back ("--std");
    args.push_back ("c++20");
    args.push_back ("-rdc");
    args.push_back ("true");
    args.push_back ("--expt-relaxed-constexpr");
    args.push_back ("--machine");
    args.push_back ("64");

    // Suppress warning 4819 and enable __cplusplus macro
    args.push_back ("-Xcompiler");
    args.push_back ("/wd 4819 /Zc:__cplusplus");

    // Set GPU architecture
    args.push_back ("--gpu-architecture");
    args.push_back (architecture);

    if (buildMode == "Debug")
    {
        args.push_back ("-D_DEBUG=1");
        // For OptiX kernels, use -lineinfo instead of -G for smaller PTX files
        if (fileName.rfind ("optix_", 0) == 0)
            args.push_back ("-lineinfo");
        else
            args.push_back ("-G");
    }

    args.push_back ("-ccbin");
    args.push_back ("C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.42.34433/bin/Hostx64/x64/");

    // Include paths
    args.push_back ("--include-path");
    args.push_back ("C:/ProgramData/NVIDIA Corporation/OptiX SDK 9.0.0/include");

    args.push_back ("--include-path");
    args.push_back ("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/include");

    args.push_back ("--include-path");
    args.push_back (shockerUtilFolder.generic_string());

    // Add the excludeFromBuild directory as include path so relative includes work
    // This allows CUDA files to find headers with paths like "engines/milo/milo_shared.h"
    std::filesystem::path excludeFromBuildPath = shockerUtilFolder.parent_path().parent_path() / "Milo" / "framework" / "claude_core" / "excludeFromBuild";
    args.push_back ("--include-path");
    args.push_back (excludeFromBuildPath.generic_string());

    args.push_back ("--output-file");
    args.push_back (outPath);

    // Run compilation
    int status = -1;
    std::error_code errCode;

    reproc::options options;
    options.redirect.parent = true;
    options.deadline = reproc::milliseconds (30000); // 30 second timeout

    std::tie (status, errCode) = reproc::run (args, options);

    if (errCode.value() != 0)
    {
        LOG (WARNING) << "Failed to compile " << fileName << ": " << errCode.message();
        return false;
    }

    return true;
}

CudaCompiler::CompilationResult CudaCompiler::compileForArchitecture (
    const std::filesystem::path& cudaFolder,
    const std::filesystem::path& outputFolder,
    const std::string& architecture,
    const std::string& buildMode,
    const std::filesystem::path& shockerUtilFolder,
    const std::string& engineFilter)
{
    CompilationResult result;

    // Get CUDA files based on engine filter
    auto cuFiles = getCudaFilesForEngine (cudaFolder, engineFilter);

    if (cuFiles.empty())
    {
        LOG (WARNING) << "No CUDA files found for engine: " << engineFilter;
        return result;
    }

    LOG (INFO) << "Processing " << cuFiles.size() << " CUDA files for " << engineFilter;

    for (const auto& cuFile : cuFiles)
    {
        std::string fileName = cuFile.filename().string();

        // Determine if this file should generate PTX
        bool ptx = false;

        // Check various prefixes to determine if PTX is needed
        if (fileName.rfind ("copy", 0) == 0 ||
            fileName.rfind ("deform", 0) == 0 ||
            fileName.rfind ("compute_light", 0) == 0 ||
            fileName.rfind ("compute_area_light", 0) == 0 ||
            fileName.rfind ("optix_", 0) == 0)
        {
            ptx = true;
        }

        // Compile the file
        bool wasCompiled = compileSingleFile (cuFile, outputFolder, architecture,
                                              buildMode, shockerUtilFolder, ptx);

        if (wasCompiled)
        {
            // Check if the output file exists to determine success
            std::string outPath = ptx
                                      ? (outputFolder / cuFile.stem()).string() + ".ptx"
                                      : (outputFolder / cuFile.stem()).string() + ".optixir";

            if (std::filesystem::exists (outPath))
            {
                result.compiled++;
            }
            else
            {
                result.failed++;
                result.failedFiles.push_back (fileName);
            }
        }
        else
        {
            result.skipped++;
        }
    }

    return result;
}

void CudaCompiler::compile (const std::filesystem::path& resourceFolder,
                            const std::filesystem::path& repoFolder,
                            const std::vector<std::string>& targetArchitectures,
                            const std::string& engineFilter)
{
    ScopedStopWatch sw (_FN_);

    std::string buildMode = "Release";
#ifndef NDEBUG
    buildMode = "Debug";
#endif

    LOG (INFO) << "===== CUDA Kernel Compilation =====";
    LOG (INFO) << "Engine: " << engineFilter;
    LOG (INFO) << "Build:  " << buildMode;
    LOG (INFO) << "Force:  " << (forceCompile_ ? "Yes" : "No");

    verifyPath (resourceFolder);

    // Create debug/release base folder in ptx directory
    std::filesystem::path baseOutputFolder = resourceFolder / "ptx" / buildMode;
    if (!std::filesystem::exists (baseOutputFolder))
    {
        std::filesystem::create_directories (baseOutputFolder);
    }

    verifyPath (repoFolder);

    std::filesystem::path cudaFolder = repoFolder / "framework" / "claude_core" / "excludeFromBuild";
    verifyPath (cudaFolder);

    std::filesystem::path shockerUtilFolder = repoFolder / "thirdparty/optiXUtil/src";
    verifyPath (shockerUtilFolder);

    CompilationResult totalResult;

    // For each target architecture, compile CUDA files
    for (const auto& arch : targetArchitectures)
    {
        if (arch != "sm_86") continue; // Currently only supporting sm_86

        LOG (INFO) << "Architecture: " << arch;

        // Create architecture-specific output folder
        std::filesystem::path archOutputFolder = baseOutputFolder / arch;
        if (!std::filesystem::exists (archOutputFolder))
        {
            std::filesystem::create_directories (archOutputFolder);
        }

        // Compile for this architecture
        auto archResult = compileForArchitecture (cudaFolder, archOutputFolder, arch,
                                                  buildMode, shockerUtilFolder, engineFilter);

        // Accumulate results
        totalResult.compiled += archResult.compiled;
        totalResult.skipped += archResult.skipped;
        totalResult.failed += archResult.failed;
        totalResult.failedFiles.insert (totalResult.failedFiles.end(),
                                        archResult.failedFiles.begin(),
                                        archResult.failedFiles.end());
    }

    // Print final summary
    totalResult.print();

    if (totalResult.hasErrors())
    {
        LOG (WARNING) << "CUDA compilation completed with errors";
    }
    else if (totalResult.compiled > 0)
    {
        LOG (INFO) << "CUDA compilation completed successfully";
    }
    else
    {
        LOG (INFO) << "All CUDA files are up-to-date";
    }
}

bool CudaCompiler::hasFolderChanged (const std::string& folderPath, const std::string& jsonFilePath, const std::string& buildMode)
{
    // This method is kept for backward compatibility but not used in the new implementation
    return true;
}

void CudaCompiler::verifyPath (const std::filesystem::path& path)
{
    if (!std::filesystem::exists (path))
        throw std::runtime_error ("Invalid path: " + path.string());
}