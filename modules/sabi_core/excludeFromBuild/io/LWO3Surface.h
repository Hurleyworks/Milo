#pragma once

// LWO3Surface: Encapsulates a LightWave Object surface, including geometry and material properties.
// - Maps surface data between LWO3 and CgModel formats
// - Manages material properties via an internal node graph
// - Provides access to texture maps through the material node connections
// - Handles both geometry (indices) and material properties in a unified way


using sabi::BSDFInput;
using sabi::ImageNodeInfo;
using sabi::LWO3NodeGraph;

class LWO3Surface
{
 public:
    // Constructs a surface with the given name
    LWO3Surface (const std::string& surfaceName) :
        name (surfaceName) {}
    ~LWO3Surface() = default;

    // Returns the surface name
    [[nodiscard]] const std::string& getName() const { return name; }

    // Access to triangle indices matrix
    [[nodiscard]] MatrixXu& indices() { return F; }
    [[nodiscard]] const MatrixXu& indices() const { return F; }

    // Returns the material node graph
    [[nodiscard]] const LWO3NodeGraph* getNodeGraph() const { return nodeGraph.get(); }

    // Returns total number of triangles
    [[nodiscard]] size_t triangleCount() const { return F.cols(); }

    // Updates the surface name
    void setName (const std::string& surfaceName) { name = surfaceName; }

    // Takes ownership of a new node graph
    void setNodeGraph (std::unique_ptr<LWO3NodeGraph> graph) { nodeGraph = std::move (graph); }

    // Finds the image node connected to a specific BSDF input socket
    std::optional<ImageNodeInfo> getBSDFImageNode (BSDFInput input) const;

 private:
    std::string name;
    MatrixXu F; // Triangle indices
    std::unique_ptr<LWO3NodeGraph> nodeGraph;
};