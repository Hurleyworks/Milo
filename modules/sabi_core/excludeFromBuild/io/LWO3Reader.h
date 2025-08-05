#pragma once

// LWO3Reader: Primary class for reading LightWave Object 3 (LWO3) files
// Handles file parsing and layer management while individual LWO3Layer instances
// manage their own geometry and surface data. Provides access to the loaded
// layers and maintains the root form structure.

class LWO3Reader
{
 public:
    // Reads and parses an LWO3 file at the given path
    bool read (const fs::path& filepath);

    // Returns all layers in the LWO3 file
    const std::vector<std::shared_ptr<LWO3Layer>>& getLayers() const { return layers_; }

    // Returns layer at specified index or nullptr if index is invalid
    std::shared_ptr<LWO3Layer> getLayerByIndex (size_t index) const;

    // Returns first layer matching the given name or nullptr if not found
    std::shared_ptr<LWO3Layer> getLayerByName (const std::string& name) const;

    // Returns true if file was successfully loaded and parsed
    bool isValid() const { return !layers_.empty() && root_ != nullptr; }

    // Returns last error message if any operation failed
    const std::string& getError() const { return errorMessage_; }

 private:
    // Creates layer objects from the parsed form structure
    bool parseLayers();

    std::shared_ptr<LWO3Form> root_;
    std::vector<std::shared_ptr<LWO3Layer>> layers_;
    std::string errorMessage_;
};