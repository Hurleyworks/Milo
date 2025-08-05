#pragma once

#include "../../engine_core.h"

class CudaCompiler
{
 public:
    CudaCompiler() = default;
    ~CudaCompiler() = default;

    // Modified to accept a vector of target architectures
    void compile (const std::filesystem::path& resourceFolder,
                  const std::filesystem::path& repoFolder,
                  const std::vector<std::string>& targetArchitectures = {"sm_75", "sm_80", "sm_86", "sm_90"});

 private:
    void verifyPath (const std::filesystem::path& path);
    bool hasFolderChanged (const std::string& folderPath, const std::string& jsonFilePath, const std::string& buildMode);

    // Helper to run compilation for a specific architecture
    void compileForArchitecture (const std::filesystem::path& cudaFolder,
                                 const std::filesystem::path& outputFolder,
                                 const std::string& architecture,
                                 const std::string& buildMode,
                                 const std::filesystem::path& shockerUtilFolder);
};