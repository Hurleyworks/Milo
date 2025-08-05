

// Attempts to read and parse the LWO3 file at the given path
bool LWO3Reader::read (const fs::path& filepath)
{
    // Clear any existing data
    root_.reset();
    layers_.clear();
    errorMessage_.clear();

    try
    {
        // Parse file into form structure
        LWO3Tree parser;
        root_ = parser.read (filepath);

        if (!root_)
        {
            errorMessage_ = "Failed to parse LWO3 file";
            return false;
        }

        // Parse layers from form structure
        if (!parseLayers())
        {
            return false;
        }

        return true;
    }
    catch (const std::exception& e)
    {
        errorMessage_ = "Exception while reading LWO3 file: " + std::string (e.what());
        return false;
    }
}

// Returns layer at specified index or nullptr if invalid
std::shared_ptr<LWO3Layer> LWO3Reader::getLayerByIndex (size_t index) const
{
    if (index < layers_.size())
    {
        return layers_[index];
    }
    return nullptr;
}

// Returns first layer matching name or nullptr if not found
std::shared_ptr<LWO3Layer> LWO3Reader::getLayerByName (const std::string& name) const
{
    for (const auto& layer : layers_)
    {
        if (layer->getName() == name)
        {
            return layer;
        }
    }
    return nullptr;
}

// Creates layer objects from the parsed form structure
bool LWO3Reader::parseLayers()
{
    if (!root_)
    {
        errorMessage_ = "No root form available";
        return false;
    }

    LWO3Navigator navigator;
    auto results = navigator.findElementsById (root_.get(), sabi::LWO::LAYR);

    if (results.empty())
    {
        errorMessage_ = "No layers found in LWO3 file";
        return false;
    }

    // Create layers in order of their indices
    size_t maxLayerIndex = 0;

    // First pass to find highest layer index
    for (const auto& result : results)
    {
        if (!result.element->isForm())
        {
            const auto* chunk = static_cast<const LWO3Chunk*> (result.element);
            BinaryReader reader (
                const_cast<char*> (reinterpret_cast<const char*> (chunk->getData().data())),
                static_cast<uint32_t> (chunk->getData().size()));

            uint16_t layerIndex = mace::swap16 (reader.ReadUint16());
            maxLayerIndex = std::max (maxLayerIndex, static_cast<size_t> (layerIndex));
        }
    }

    // Resize layers vector to accommodate all indices
    layers_.resize (maxLayerIndex + 1);

    // Create layers
    for (const auto& result : results)
    {
        if (!result.element->isForm())
        {
            const auto* chunk = static_cast<const LWO3Chunk*> (result.element);
            BinaryReader reader (
                const_cast<char*> (reinterpret_cast<const char*> (chunk->getData().data())),
                static_cast<uint32_t> (chunk->getData().size()));

            uint16_t layerIndex = mace::swap16 (reader.ReadUint16());

            try
            {
                auto layer = std::make_shared<LWO3Layer> (root_, layerIndex);
                layers_[layerIndex] = layer;
            }
            catch (const std::exception& e)
            {
                LOG (WARNING) << "Failed to create layer " << layerIndex << ": " << e.what();
                return false;
            }
        }
    }

    // Remove any null layers from gaps in indices
    layers_.erase (
        std::remove_if (layers_.begin(), layers_.end(),
                        [] (const auto& layer)
                        { return layer == nullptr; }),
        layers_.end());

    return !layers_.empty();
}