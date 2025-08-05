// LWO3Material: Represents a single material from an LWO3 file including its node network
// Extracts and manages material properties from both Standard and Principled BSDF nodes
// Provides access to connected texture maps and material parameters
// Supports both legacy Standard material nodes and modern physically-based BSDF nodes

#pragma once

class LWO3Material
{
 public:
    // Initialize material with surface name
    LWO3Material (const std::string& name, std::shared_ptr<LWO3Form> root, const LWO3Form* surfForm);

    const std::string& getName() const { return name_; }
    const std::vector<ImageNodeInfo>& getImageNodes() const { return imageNodes_; }
    const std::vector<PrincipledBSDFInfo>& getBSDFNodes() const { return bsdfNodes_; }
    const std::vector<StandardNodeInfo>& getStandardNodes() const { return standardNodes_; }

    // Gets image node connected to specific BSDF input
    std::optional<ImageNodeInfo> getBSDFImageNode (BSDFInput input) const;

    // Gets whether this material uses Standard or BSDF nodes
    bool usesStandardNodes() const { return !standardNodes_.empty(); }
    bool usesBSDFNodes() const { return !bsdfNodes_.empty(); }

    const LWO3Form* getSurfaceForm() const { return surfForm_; }
    float getMaxSmoothingAngle() const;

 private:
    std::string name_;
    const LWO3Form* surfForm_;
    std::vector<ImageNodeInfo> imageNodes_;
    std::vector<PrincipledBSDFInfo> bsdfNodes_;
    std::vector<StandardNodeInfo> standardNodes_;
    std::vector<NodeConnection> nodeConnections_;
};