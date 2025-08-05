#pragma once

// LWO3Layer: Represents a layer within a LightWave Object
// Layers are the primary organizational structure in LWO3 files containing
// geometry, surfaces, and transformational data. Each layer has its own
// coordinate space defined by its pivot point and can contain multiple
// surfaces with associated vertices and polygons.

using Eigen::Affine3f;
using Eigen::AlignedBox3f;
using Eigen::Quaternionf;
using Eigen::Vector3f;

using sabi::BSDFInput;
using sabi::LWO3Navigator;
using sabi::LWO3NodeGraph;
using sabi::LWO3Surface;
using sabi::NodeConnection;

// Structure to represent a polygon
struct LWPolygon
{
    std::vector<uint32_t> indices; // Vertex indices
    uint16_t flags = 0;            // High 6 bits from vertex count
};

// Structure to hold vertex map data
struct VertexMap
{
    uint32_t type;                       // TXUV, WGHT, etc
    uint16_t dimension;                  // Number of values per vertex
    std::string name;                    // Map name
    std::vector<uint32_t> vertexIndices; // Indices of mapped vertices
    std::vector<float> values;           // Dimension * vertexIndices.size() values
};

class LWO3Layer
{
 public:
    // Constructs a layer from a LAYR chunk in the LWO3 form
    LWO3Layer (std::shared_ptr<LWO3Form> root, size_t layerIndex);

    // Gets the layer's index number
    size_t getIndex() const { return index_; }

    // Gets the layer's flags (bit 0 = visibility)
    uint16_t getFlags() const { return flags_; }

    // Gets the layer's pivot point
    const Vector3f& getPivot() const { return pivot_; }

    // Gets the layer's name
    const std::string& getName() const { return name_; }

    // Gets the parent layer index (-1 if no parent)
    int16_t getParentIndex() const { return parentIndex_; }

    // Should return false if bit 0 is set (hidden), true otherwise
    bool isVisible() const { return (flags_ & 1) == 0; }

    // Gets the layer's points/vertices
    const std::vector<Vector3f>& getPoints() const { return points_; }

    // Get the layer's bounding box
    const AlignedBox3f& getBoundingBox() const { return bbox_; }

    // Get polygon type (FACE, PTCH, etc)
    uint32_t getPolygonType() const { return polygonType_; }

    // Get all polygons
    const std::vector<LWPolygon>& getPolygons() const { return polygons_; }

    // Check if polygons are of specific type
    bool isType (uint32_t type) const { return polygonType_ == type; }

    // Get all surface tags
    const std::vector<std::string>& getTags() const { return tags_; }

    // Get tag at specific index
    const std::string& getTag (size_t index) const
    {
        return index < tags_.size() ? tags_[index] : emptyTag_;
    }

    // Get surface tag index for a polygon
    int getPolygonTagIndex (uint32_t polyIndex, uint32_t tagType) const
    {
        auto it = polyTags_.find (tagType);
        if (it != polyTags_.end() && polyIndex < it->second.size())
        {
            return it->second[polyIndex];
        }
        return -1;
    }

    // Get surface tag name for a polygon
    std::string getPolygonTag (uint32_t polyIndex, uint32_t tagType) const
    {
        int tagIndex = getPolygonTagIndex (polyIndex, tagType);
        if (tagIndex >= 0 && tagIndex < tags_.size())
        {
            return tags_[tagIndex];
        }
        return "";
    }

    // Gets all vertex maps of a specific type
    std::vector<const VertexMap*> getVertexMaps (uint32_t type) const;

    // Gets a vertex map by name
    const VertexMap* getVertexMapByName (const std::string& name) const;

    // Gets UV coordinates for a vertex if they exist
    bool getVertexUV (uint32_t vertexIndex, float& u, float& v) const;

    // Get all surfaces used by this layer
    const std::vector<std::shared_ptr<LWO3Surface>>& getSurfaces() const { return surfaces_; }

    // Get surface by name
    LWO3Surface* getSurfaceByName (const std::string& name);
    const LWO3Surface* getSurfaceByName (const std::string& name) const;

    // Get surface used by a specific polygon
    LWO3Surface* getPolygonSurface (size_t polyIndex);
    const LWO3Surface* getPolygonSurface (size_t polyIndex) const;

 private:
    LWO3Navigator navigator_;
    std::shared_ptr<LWO3Form> root_;

    size_t index_ = 0;
    uint16_t flags_ = 0;
    Vector3f pivot_ = Vector3f::Zero();
    std::string name_;
    int16_t parentIndex_ = -1;

    uint32_t polygonType_ = 0;        // Type of polygons (FACE, PTCH, etc)
    std::vector<LWPolygon> polygons_; // List of polygons

    std::vector<std::string> tags_; // List of surface tags
    const std::string emptyTag_;    // Empty string for invalid indices

    // Map of tag type to vector of tag indices
    // Key is tag type (SURF, PART, etc)
    // Value is vector of indices into tags_ array, one per polygon
    std::map<uint32_t, std::vector<uint16_t>> polyTags_;

    std::vector<Vector3f> points_;
    AlignedBox3f bbox_; // Stores min/max corners of bounding box

    std::vector<VertexMap> vertexMaps_;

    std::vector<std::shared_ptr<LWO3Surface>> surfaces_;

    // Chunk processing functions
    bool processLayrChunk (const LWO3Chunk* chunk);
    bool processPntsChunk (const LWO3Chunk* chunk);
    bool processBBoxChunk (const LWO3Chunk* chunk);
    bool processPolsChunk (const LWO3Chunk* chunk);
    bool processTagsChunk (const LWO3Chunk* chunk);
    bool processPTagChunk (const LWO3Chunk* chunk);
    bool processVMapChunk (const LWO3Chunk* chunk);
    bool processSurfaceForms();

    // Helper to find specific chunk
    const LWO3Chunk* findChunk (uint32_t chunkId, size_t startOffset = 0);
};