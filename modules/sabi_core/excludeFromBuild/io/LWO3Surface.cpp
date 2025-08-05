

// Retrieves image node connected to specified BSDF input socket
// Returns nullopt if no connection found
std::optional<ImageNodeInfo> LWO3Surface::getBSDFImageNode (BSDFInput input) const
{
    if (!nodeGraph) return std::nullopt;

    std::string inputName = getBSDFInputName (input);
    for (const auto& img : nodeGraph->getImageNodes())
    {
        for (const auto& conn : img.connections)
        {
            if (conn.targetNode.find ("Principled BSDF") != std::string::npos &&
                conn.inputName == inputName)
            {
                return img;
            }
        }
    }
    return std::nullopt;
}