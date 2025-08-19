#pragma once

#include "../../milo_core.h"

class CudaCompiler
{
 public:
    // Compilation statistics
    struct CompilationResult
    {
        int compiled = 0;
        int skipped = 0;
        int failed = 0;
        std::vector<std::string> failedFiles;
        
        void print() const;
        bool hasErrors() const { return failed > 0; }
    };

    CudaCompiler() = default;
    ~CudaCompiler() = default;

    // Modified to accept a vector of target architectures and engine filter
    void compile (const std::filesystem::path& resourceFolder,
                  const std::filesystem::path& repoFolder,
                  const std::vector<std::string>& targetArchitectures = {"sm_75", "sm_80", "sm_86", "sm_90"},
                  const std::string& engineFilter = "all");
    
    // Force recompilation of all files
    void setForceCompile(bool force) { forceCompile_ = force; }

 private:
    void verifyPath (const std::filesystem::path& path);
    bool hasFolderChanged (const std::string& folderPath, const std::string& jsonFilePath, const std::string& buildMode);
    
    // Check if a specific file needs compilation
    bool needsCompilation(const std::filesystem::path& cuPath, 
                         const std::filesystem::path& ptxPath);

    // Helper to run compilation for a specific architecture
    CompilationResult compileForArchitecture (const std::filesystem::path& cudaFolder,
                                             const std::filesystem::path& outputFolder,
                                             const std::string& architecture,
                                             const std::string& buildMode,
                                             const std::filesystem::path& shockerUtilFolder,
                                             const std::string& engineFilter);
    
    // Compile a single CUDA file
    bool compileSingleFile(const std::filesystem::path& cuFile,
                          const std::filesystem::path& outputFolder,
                          const std::string& architecture,
                          const std::string& buildMode,
                          const std::filesystem::path& shockerUtilFolder,
                          bool generatePTX);
    
    // Get CUDA files for specific engine
    std::vector<std::filesystem::path> getCudaFilesForEngine(const std::filesystem::path& baseFolder,
                                                            const std::string& engineFilter);
    
    bool forceCompile_ = false;
};