#pragma once

//#include "LWO3Material.h"

// Manages extraction and access to all materials from an LWO3 file
class LWO3MaterialManager
{
 public:
    explicit LWO3MaterialManager (std::shared_ptr<LWO3Form> root);

    // Get material by surface name
    const LWO3Material* getMaterial (const std::string& surfaceName) const;

    // Get all materials
    const std::vector<std::shared_ptr<LWO3Material>>& getMaterials() const
    {
        return materials_;
    }

 private:
    std::vector<std::shared_ptr<LWO3Material>> materials_;
    void extractMaterials (std::shared_ptr<LWO3Form> root);
};