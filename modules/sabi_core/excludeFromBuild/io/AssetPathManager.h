
#pragma once

//#include <mace_core/mace_core.h>

class AssetPathManager
{
 public:
    void addAssetMapping (const fs::path& originalPath, const fs::path& newRelativePath)
    {
        assetMap[originalPath.string()] = newRelativePath.string();
    }

    std::string getRelativePath (const std::string& originalPath) const
    {
        auto it = assetMap.find (originalPath);
        if (it != assetMap.end())
        {
            return it->second;
        }
        return originalPath; // Return original path if not found
    }

 private:
    std::unordered_map<std::string, std::string> assetMap;
};