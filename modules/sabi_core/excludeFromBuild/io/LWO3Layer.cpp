

LWO3Layer::LWO3Layer (std::shared_ptr<LWO3Form> root, size_t layerIndex) :
    root_ (root), 
    index_ (layerIndex)
{
    if (!root)
    {
        LOG (INFO) << "Null root form provided to LWO3Layer";
        return;
    }

    // Find and process TAGS chunk first since other chunks may reference tags
    if (const LWO3Chunk* tagsChunk = findChunk (sabi::LWO::TAGS))
    {
        if (!processTagsChunk (tagsChunk))
        {
            LOG (WARNING) << "Failed to process TAGS chunk for layer " << index_;
        }
    }

    // Find and process LAYR chunk
    if (const LWO3Chunk* layrChunk = findChunk ( sabi::LWO::LAYR))
    {
        if (!processLayrChunk (layrChunk))
        {
            LOG (WARNING) << "Failed to process LAYR chunk for index " << index_;
            return;
        }
    }

    // Find and process PNTS chunk
    if (const LWO3Chunk* pntsChunk = findChunk (sabi::LWO::PNTS))
    {
        if (!processPntsChunk (pntsChunk))
        {
            LOG (WARNING) << "Failed to process PNTS chunk for layer " << index_;
        }
    }

    // Find and process BBOX chunk
    if (const LWO3Chunk* bboxChunk = findChunk ( sabi::LWO::BBOX))
    {
        if (!processBBoxChunk (bboxChunk))
        {
            LOG (WARNING) << "Failed to process BBOX chunk for layer " << index_;
        }
    }

    // Find and process POLS chunk
    if (const LWO3Chunk* polsChunk = findChunk (sabi::LWO::POLS))
    {
        if (!processPolsChunk (polsChunk))
        {
            LOG (WARNING) << "Failed to process POLS chunk for layer " << index_;
        }
    }

    if (!processSurfaceForms ())
    {
        LOG (WARNING) << "Failed to process SURF forms for layer " << index_;
    }

    // Process PTAG chunks after POLS and TAGS
    if (const LWO3Chunk* ptagChunk = findChunk (sabi::LWO::PTAG))
    {
        if (!processPTagChunk (ptagChunk))
        {
            LOG (WARNING) << "Failed to process PTAG chunk for layer " << index_;
        }
    }

    //  after other chunk processing
    if (const LWO3Chunk* vmapChunk = findChunk (sabi::LWO::VMAP))
    {
        if (!processVMapChunk (vmapChunk))
        {
            LOG (WARNING) << "Failed to process VMAP chunk for layer " << index_;
        }
    }
}

bool LWO3Layer::processBBoxChunk (const LWO3Chunk* chunk)
{
    // BBOX chunk contains two Vector3f (24 bytes total)
    if (chunk->getData().size() != 24)
    {
        LOG (WARNING) << "Invalid BBOX chunk size: " << chunk->getData().size();
        return false;
    }

    BinaryReader reader (
        const_cast<char*> (reinterpret_cast<const char*> (chunk->getData().data())),
        static_cast<uint32_t> (chunk->getData().size()));

    // Read min point
    float minX = mace::swapFloat (reader.ReadFloat());
    float minY = mace::swapFloat (reader.ReadFloat());
    float minZ = mace::swapFloat (reader.ReadFloat());

    // Read max point
    float maxX = mace::swapFloat (reader.ReadFloat());
    float maxY = mace::swapFloat (reader.ReadFloat());
    float maxZ = mace::swapFloat (reader.ReadFloat());

    bbox_.min() = Vector3f (minX, minY, minZ);
    bbox_.max() = Vector3f (maxX, maxY, maxZ);

    return true;
}

bool LWO3Layer::processPolsChunk (const LWO3Chunk* chunk)
{
    BinaryReader reader (
        const_cast<char*> (reinterpret_cast<const char*> (chunk->getData().data())),
        static_cast<uint32_t> (chunk->getData().size()));

    // Read polygon type (FACE, PTCH, etc)
    polygonType_ = mace::swap32 (reader.ReadUint32());

    // Read polygons until we reach the end of the chunk
    while (reader.Position() < reader.Length())
    {
        // Read vertex count including flags
        uint16_t vertCountAndFlags = mace::swap16 (reader.ReadUint16());

        // Extract count and flags
        uint16_t flags = vertCountAndFlags & 0xFC00;     // High 6 bits
        uint16_t vertCount = vertCountAndFlags & 0x03FF; // Low 10 bits

        // Create new polygon
        LWPolygon poly;
        poly.flags = flags;
        poly.indices.reserve (vertCount);

        // Read vertex indices
        for (uint16_t i = 0; i < vertCount; i++)
        {
            // Use readVX to handle variable-length indices
            uint32_t index = mace::readVX (reader);
            poly.indices.push_back (index);
        }

        polygons_.push_back (std::move (poly));
    }

    return true;
}
const LWO3Chunk* LWO3Layer::findChunk (uint32_t chunkId, size_t startOffset)
{
    auto results = navigator_.findElementsById (root_.get(), chunkId);

    for (const auto& result : results)
    {
        if (!result.element->isForm())
        {
            const auto* chunk = static_cast<const LWO3Chunk*> (result.element);

            // For LAYR chunks, verify it's for our layer
            if (chunkId == sabi::LWO::LAYR)
            {
                BinaryReader reader (
                    const_cast<char*> (reinterpret_cast<const char*> (chunk->getData().data())),
                    static_cast<uint32_t> (chunk->getData().size()));

                uint16_t layerNumber = mace::swap16 (reader.ReadUint16());
                if (layerNumber == index_)
                {
                    return chunk;
                }
            }
            else
            {
                if (chunk->getFileOffset() >= startOffset)
                {
                    return chunk;
                }
            }
        }
    }
    return nullptr;
}

bool LWO3Layer::processLayrChunk (const LWO3Chunk* chunk)
{
    BinaryReader reader (
        const_cast<char*> (reinterpret_cast<const char*> (chunk->getData().data())),
        static_cast<uint32_t> (chunk->getData().size()));

    // Skip layer number since we already verified it
    reader.Skip (2);

    // Read flags
    flags_ = mace::swap16 (reader.ReadUint16());

    // Read pivot point
    float x = mace::swapFloat (reader.ReadFloat());
    float y = mace::swapFloat (reader.ReadFloat());
    float z = mace::swapFloat (reader.ReadFloat());
    pivot_ = Vector3f (x, y, z);

    // Read name
    name_ = reader.ReadNullTerminatedString();

    // Read parent index if there are remaining bytes
    if (reader.Position() < reader.Length())
    {
        parentIndex_ = mace::swap16 (reader.ReadInt16());
    }

    return true;
}

bool LWO3Layer::processPntsChunk (const LWO3Chunk* chunk)
{
    const auto& data = chunk->getData();
    BinaryReader reader (
        const_cast<char*> (reinterpret_cast<const char*> (data.data())),
        static_cast<uint32_t> (data.size()));

    // Each point is 12 bytes (3 x 4-byte floats)
    size_t numPoints = data.size() / 12;
    points_.reserve (numPoints);

    for (size_t i = 0; i < numPoints; i++)
    {
        float x = mace::swapFloat (reader.ReadFloat());
        float y = mace::swapFloat (reader.ReadFloat());
        float z = mace::swapFloat (reader.ReadFloat());
        points_.emplace_back (x, y, z);
    }

    return true;
}

bool LWO3Layer::processTagsChunk (const LWO3Chunk* chunk)
{
    BinaryReader reader (
        const_cast<char*> (reinterpret_cast<const char*> (chunk->getData().data())),
        static_cast<uint32_t> (chunk->getData().size()));

    // Read strings until we reach end of chunk
    while (reader.Position() < reader.Length())
    {
        // Read null-terminated string
        std::string tag = reader.ReadNullTerminatedString();
        if (!tag.empty())
        {
            tags_.push_back (tag);
        }
        else
        {
            LOG (WARNING) << "Empty tag found in TAGS chunk";
        }

        // Handle padding - strings must be aligned to 2 bytes
        if (reader.Position() % 2 != 0)
        {
            reader.Skip (1);
        }
    }

    return !tags_.empty();
}

bool LWO3Layer::processPTagChunk (const LWO3Chunk* chunk)
{
    BinaryReader reader (
        const_cast<char*> (reinterpret_cast<const char*> (chunk->getData().data())),
        static_cast<uint32_t> (chunk->getData().size()));

    // Read tag type (SURF, PART, etc)
    uint32_t tagType = mace::swap32 (reader.ReadUint32());

    // Create or get vector of tag indices for this type
    auto& tagIndices = polyTags_[tagType];

    // Ensure vector is sized to match number of polygons
    if (tagIndices.empty() && !polygons_.empty())
    {
        tagIndices.resize (polygons_.size(), 0);
    }

    // Read polygon/tag pairs until end of chunk
    while (reader.Position() < reader.Length())
    {
        // Read polygon index using variable-length format
        uint32_t polyIndex = mace::readVX (reader);

        // Read tag index
        uint16_t tagIndex = mace::swap16 (reader.ReadUint16());

        // Validate indices
        if (polyIndex >= polygons_.size())
        {
            LOG (WARNING) << "Invalid polygon index in PTAG chunk: " << polyIndex;
            continue;
        }

        if (tagType == sabi::LWO::SURF && tagIndex >= tags_.size())
        {
            LOG (WARNING) << "Invalid tag index in PTAG chunk: " << tagIndex;
            continue;
        }

        // Store tag index for this polygon
        tagIndices[polyIndex] = tagIndex;
    }

    return true;
}

bool LWO3Layer::processVMapChunk (const LWO3Chunk* chunk)
{
    BinaryReader reader (
        const_cast<char*> (reinterpret_cast<const char*> (chunk->getData().data())),
        static_cast<uint32_t> (chunk->getData().size()));

    // Read map type (TXUV, WGHT, etc)
    uint32_t type = mace::swap32 (reader.ReadUint32());

    // Read dimension
    uint16_t dimension = mace::swap16 (reader.ReadUint16());

    // Read name
    std::string name = reader.ReadNullTerminatedString();

    
    // Skip any padding after name string to align to next vertex index
    reader.Align (2);


    // Create new vertex map
    VertexMap vmap;
    vmap.type = type;
    vmap.dimension = dimension;
    vmap.name = name;

    // Read vertex/value pairs until end of chunk
    while (reader.Position() < reader.Length())
    {
        // Read vertex index
        uint32_t vertIndex = mace::readVX (reader);
        vmap.vertexIndices.push_back (vertIndex);

        // Read dimension number of values
        for (uint16_t i = 0; i < dimension; i++)
        {
            float value = mace::swapFloat (reader.ReadFloat());
            // flip the V to match LW
            // lost hours trying to track this
            if (i == 1)
               value = 1.0f - value;
            vmap.values.push_back (value);
        }
    }

    vertexMaps_.push_back (std::move (vmap));
   
    return true;
}

std::vector<const VertexMap*> LWO3Layer::getVertexMaps (uint32_t type) const
{
    std::vector<const VertexMap*> maps;
    for (const auto& map : vertexMaps_)
    {
        if (map.type == type)
        {
            maps.push_back (&map);
        }
    }
    return maps;
}

const VertexMap* LWO3Layer::getVertexMapByName (const std::string& name) const
{
    for (const auto& map : vertexMaps_)
    {
        if (map.name == name)
        {
            return &map;
        }
    }
    return nullptr;
}

bool LWO3Layer::getVertexUV (uint32_t vertexIndex, float& u, float& v) const
{
    // Look for TXUV maps
    for (const auto& map : vertexMaps_)
    {
        if (map.type == sabi::LWO::TXUV && map.dimension == 2)
        {
            // Find vertex in this map
            for (size_t i = 0; i < map.vertexIndices.size(); i++)
            {
                if (map.vertexIndices[i] == vertexIndex)
                {
                    // Get UV values
                    u = map.values[i * 2];
                    v = map.values[i * 2 + 1];
                    return true;
                }
            }
        }
    }
    return false;
}

bool LWO3Layer::processSurfaceForms()
{
    if (!root_)
    {
        return false;
    }

    auto results = navigator_.findElementsById (root_.get(), sabi::LWO::SURF);
    if (results.empty())
    {
        LOG (WARNING) << "No SURF forms found";
        return false;
    }

    for (const auto& result : results)
    {
        if (!result.element->isForm())
        {
            continue;
        }

        const auto* surfForm = static_cast<const LWO3Form*> (result.element);

        std::string surfaceName;
        std::string parentName;

        for (const auto& child : surfForm->getChildren())
        {
            if (!child->isForm() && child->getId() == sabi::LWO::ANON)
            {
                BinaryReader reader (
                    const_cast<char*> (reinterpret_cast<const char*> (
                        static_cast<const LWO3Chunk*> (child.get())->getData().data())),
                    static_cast<uint32_t> (static_cast<const LWO3Chunk*> (child.get())->getData().size()));

                surfaceName = reader.ReadNullTerminatedString();
                if (reader.Position() < reader.Length())
                {
                    parentName = reader.ReadNullTerminatedString();
                }
                break;
            }
        }

        if (surfaceName.empty())
        {
            LOG (WARNING) << "Invalid SURF form - missing name";
            continue;
        }

        // Create surface
        auto surface = std::make_shared<LWO3Surface> (surfaceName);

        // Pass shared root form to node graph
        surface->setNodeGraph (std::make_unique<LWO3NodeGraph> (root_));

        surfaces_.push_back (surface);
    }

    return !surfaces_.empty();
}


LWO3Surface* LWO3Layer::getSurfaceByName (const std::string& name)
{
    for (auto& surf : surfaces_)
    {
        if (surf->getName() == name)
        {
            return surf.get();
        }
    }
    return nullptr;
}

const LWO3Surface* LWO3Layer::getSurfaceByName (const std::string& name) const
{
    for (const auto& surf : surfaces_)
    {
        if (surf->getName() == name)
        {
            return surf.get();
        }
    }
    return nullptr;
}

LWO3Surface* LWO3Layer::getPolygonSurface (size_t polyIndex)
{
    // Get surface tag index for this polygon
    int tagIndex = getPolygonTagIndex (polyIndex, sabi::LWO::SURF);
    if (tagIndex < 0 || tagIndex >= tags_.size())
    {
        return nullptr;
    }

    // Get surface name from tag
    const std::string& surfaceName = tags_[tagIndex];
    return getSurfaceByName (surfaceName);
}

const LWO3Surface* LWO3Layer::getPolygonSurface (size_t polyIndex) const
{
    // Get surface tag index for this polygon
    int tagIndex = getPolygonTagIndex (polyIndex, sabi::LWO::SURF);
    if (tagIndex < 0 || tagIndex >= tags_.size())
    {
        return nullptr;
    }

    // Get surface name from tag
    const std::string& surfaceName = tags_[tagIndex];
    return getSurfaceByName (surfaceName);
}