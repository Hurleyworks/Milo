

LWO3Material::LWO3Material (const std::string& name, std::shared_ptr<LWO3Form> root, const LWO3Form* surfForm) :
    name_ (name),
    surfForm_ (surfForm)
{
    if (!root || !surfForm)
    {
        LOG (WARNING) << "Null form provided to LWO3Material: " << name;
        return;
    }

    // Create temporary node graph to extract node data
    LWO3NodeGraph nodeGraph (root, surfForm);

    // Extract image nodes
    imageNodes_ = nodeGraph.getImageNodes();

    // Extract BSDF nodes
    bsdfNodes_ = nodeGraph.getPrincipledBSDFNodes();

    // Extract Standard nodes
    standardNodes_ = nodeGraph.getStandardNodes();

    // Get all node connections
    for (const auto& node : nodeGraph.getNodesByType())
    {
        for (const auto& nodeName : node.second)
        {
            auto connections = nodeGraph.getNodeConnections (nodeName);
            nodeConnections_.insert (nodeConnections_.end(),
                                     connections.begin(),
                                     connections.end());
        }
    }

    //LOG (DBUG) << "Created material " << name
    //           << " with " << imageNodes_.size() << " image nodes, "
    //           << bsdfNodes_.size() << " BSDF nodes, "
    //           << standardNodes_.size() << " Standard nodes";
}

// Gets image node connected to specific BSDF input socket if one exists
std::optional<ImageNodeInfo> LWO3Material::getBSDFImageNode (BSDFInput input) const
{
    std::string inputName = getBSDFInputName (input);

    // Search all image nodes for connection to this BSDF input
    for (const auto& img : imageNodes_)
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

float LWO3Material::getMaxSmoothingAngle() const
{
    if (!surfForm_)
    {
        return 0.0f;
    }

    for (const auto& child : surfForm_->getChildren())
    {
        if (!child->isForm() && child->getId() == LWO::SMAN)
        {
            const auto* chunk = static_cast<const LWO3Chunk*> (child.get());
            if (chunk->getData().size() >= 4)
            {
                BinaryReader reader (
                    const_cast<char*> (reinterpret_cast<const char*> (chunk->getData().data())),
                    static_cast<uint32_t> (chunk->getData().size()));

                float angle = mace::swapFloat (reader.ReadFloat());
               /* LOG (DBUG) << "Found max smoothing angle for material '" << name_
                           << "': " << angle << " radians";*/
                return angle;
            }
        }
    }

    //LOG (DBUG) << "No smoothing angle found for material '" << name_ << "', using default of 0.0";
    return 0.0f;
}
