
#include "CudaCompiler.h"
#include <reproc++/run.hpp>

bool CudaCompiler::hasFolderChanged (const std::string& folderPath, const std::string& jsonFilePath, const std::string& buildMode)
{
    nlohmann::json jsonFile;
    std::ifstream inFile (jsonFilePath);

    if (inFile.is_open())
    {
        inFile >> jsonFile;
        inFile.close();
    }
    else
    {
        jsonFile = nlohmann::json::object();
    }

    bool changed = false;

    // Check if build mode has changed
    if (jsonFile.find ("buildMode") == jsonFile.end() || jsonFile["buildMode"] != buildMode)
    {
        changed = true;
        jsonFile["buildMode"] = buildMode;
    }

    for (const auto& entry : std::filesystem::directory_iterator (folderPath))
    {
        auto path = entry.path();

        // Only consider *.cu files
        if (path.extension() == ".cu")
        {
            auto pathStr = path.string();
            auto lastWriteTime = std::filesystem::last_write_time (path);
            auto timeStr = lastWriteTime.time_since_epoch().count();

            if (jsonFile.find (pathStr) == jsonFile.end() || jsonFile[pathStr] != timeStr)
            {
                changed = true;
                jsonFile[pathStr] = timeStr;
            }
        }
    }

    std::ofstream outFile (jsonFilePath);
    if (outFile.is_open())
    {
        outFile << jsonFile.dump (4);
        outFile.close();
    }

    return changed;
}

void CudaCompiler::compile (const std::filesystem::path& resourceFolder,
                            const std::filesystem::path& repoFolder,
                            const std::vector<std::string>& targetArchitectures)
{
    ScopedStopWatch sw (_FN_);

    std::string buildMode = "Release";
#ifndef NDEBUG
    buildMode = "Debug";
#endif

    verifyPath (resourceFolder);

    // Create debug/release base folder in ptx directory
    std::filesystem::path baseOutputFolder = resourceFolder / "ptx" / buildMode;
    if (!std::filesystem::exists (baseOutputFolder))
    {
        std::filesystem::create_directories (baseOutputFolder);
    }

    verifyPath (repoFolder);

    std::filesystem::path cudaFolder = repoFolder / "framework" / "engine_core" / "excludeFromBuild" / "cuda";
    verifyPath (cudaFolder);

    // Nothing to do if the cu files or build mode haven't been changed
    std::string jsonPath = baseOutputFolder.string() + "/file_times.json";
    if (!hasFolderChanged (cudaFolder.string(), jsonPath, buildMode)) return;

    std::filesystem::path shockerUtilFolder = repoFolder / "thirdparty/optiXUtil/src";
    verifyPath (shockerUtilFolder);

    // For each target architecture, compile all CUDA files
    for (const auto& arch : targetArchitectures)
    {
        if (arch != "sm_86") continue;

        LOG (DBUG) << "Compiling for architecture: " << arch;

        // Create architecture-specific output folder
        std::filesystem::path archOutputFolder = baseOutputFolder / arch;
        if (!std::filesystem::exists (archOutputFolder))
        {
            std::filesystem::create_directories (archOutputFolder);
        }

        // Compile all files for this architecture
        // Use compute_86 for PTX generation instead of sm_86
        // Compile all files for this architecture
        compileForArchitecture (cudaFolder, archOutputFolder, arch, buildMode, shockerUtilFolder);
      //  std::string computeArch = "compute_86";
       // compileForArchitecture (cudaFolder, archOutputFolder, computeArch, buildMode, shockerUtilFolder);
    }
}

void CudaCompiler::compileForArchitecture (const std::filesystem::path& cudaFolder,
                                           const std::filesystem::path& outputFolder,
                                           const std::string& architecture,
                                           const std::string& buildMode,
                                           const std::filesystem::path& shockerUtilFolder)
{
    // Path to nvcc exe
    std::string exe = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/bin/nvcc.exe";

    std::string ext = ".cu";
    std::vector<std::filesystem::path> cuFiles = FileServices::findFilesWithExtension (cudaFolder, ext);

    for (const auto& f : cuFiles)
    {
        std::string fileName = f.filename().string();

        LOG (DBUG) << "Compiling " << fileName << " for " << architecture;
        bool ptx = false;

        // copy_buffers.cu has to be ptx
        if (fileName.rfind ("copy", 0) == 0)
            ptx = true;

        // deform.cu has to be ptx
        if (fileName.rfind ("deform", 0) == 0)
            ptx = true;

        // compute_light_prob.cu has to be ptx
        if (fileName.rfind ("compute_light", 0) == 0)
            ptx = true;

        // optix_gbuffer_.cu has to be ptx or won't compile
        if (fileName.rfind ("optix_gbuffer_", 0) == 0)
            ptx = true;

        // optix_pathtracing_has to be ptx or won't compile
        if (fileName.rfind ("optix_pathtracing_", 0) == 0)
            ptx = true;

        if (fileName.rfind ("optix_basic_", 0) == 0)
            ptx = true;

        if (fileName.rfind ("optix_enviro_", 0) == 0)
            ptx = true;

        if (fileName.rfind ("optix_denoiser_", 0) == 0)
            ptx = true;

        if (fileName.rfind ("optix_geometry_", 0) == 0)
            ptx = true;

        if (fileName.rfind ("optix_triangle_", 0) == 0)
            ptx = true;

        if (fileName.rfind ("optix_shocker_", 0) == 0)
            ptx = true;

        if (fileName.rfind ("optix_tracer_", 0) == 0)
            ptx = true;

        if (fileName.rfind ("optix_scene_", 0) == 0)
            ptx = true;

        if (fileName.rfind ("optix_pick_", 0) == 0)
            ptx = true;

        if (fileName.rfind ("optix_ripr_", 0) == 0)
            ptx = true;

        if (fileName.rfind ("optix_milo_", 0) == 0)
            ptx = true;

        if (fileName.rfind ("optix_milopick_", 0) == 0)
            ptx = true;

         if (fileName.rfind ("optix_gbuffer_", 0) == 0)
            ptx = true;

          if (fileName.rfind ("optix_deform_", 0) == 0)
             ptx = true;

         if (fileName.rfind ("optix_copybuffers", 0) == 0)
             ptx = true;

         if (fileName.rfind ("compute_light_probs", 0) == 0)
             ptx = true;


        // nvcc args
        std::vector<std::string> args;

        // Path to nvcc exe
        args.push_back (exe);
        args.push_back (f.string());

        if (ptx)
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

        // Set GPU architecture - now using the parameter
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

        // if desktop
        args.push_back ("C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.42.34433/bin/Hostx64/x64/");
        // if laptop
        // args.push_back ("C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.39.33519/bin/Hostx64/x64/");

        // OptiX 9 headers
        args.push_back ("--include-path");
        args.push_back ("C:/ProgramData/NVIDIA Corporation/OptiX SDK 9.0.0/include");

        // cuda 12.9 headers
        args.push_back ("--include-path");
        args.push_back ("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/include");

        // OptixUtil
        args.push_back ("--include-path");
        args.push_back (shockerUtilFolder.generic_string());

        args.push_back ("--output-file");

        // Output to architecture-specific folder
        std::string outPath = ptx
                                  ? (outputFolder / f.stem()).string() + ".ptx"
                                  : (outputFolder / f.stem()).string() + ".optixir";

        LOG (DBUG) << "Output: " << outPath;
        args.push_back (outPath);

        int status = -1;
        std::error_code errCode;

        reproc::options options;
        options.redirect.parent = true;
        options.deadline = reproc::milliseconds (5000);

        std::tie (status, errCode) = reproc::run (args, options);

        if (errCode.value() != 0)
        {
            LOG (WARNING) << "Error compiling for " << architecture << ": " << errCode.message();
        }
    }
}
void CudaCompiler::verifyPath (const std::filesystem::path& path)
{
    if (!std::filesystem::exists (path))
        throw std::runtime_error ("Invalid path: " + path.string());
}