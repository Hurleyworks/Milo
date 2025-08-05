//#include "LWO3NodeGraph.h" 
//#include "LWO3Navigator.h"

LWO3NodeGraph::LWO3NodeGraph (std::shared_ptr<LWO3Form> root, const LWO3Form* surfForm) :
    root_ (root),
    surfForm_ (surfForm),
    surfaceName_()
{
    if (!root_ || !surfForm_)
    {
        // LOG (WARNING) << "Null form provided to LWO3NodeGraph";
        return;
    }
    extractSurfaceName();
    extractNodeNames();
}



void LWO3NodeGraph::extractSurfaceName()
{
    // Surface name is in the ANON chunk of the surfForm
    for (const auto& child : surfForm_->getChildren())
    {
        if (!child->isForm() && child->getId() == LWO::ANON)
        {
            const auto* chunk = static_cast<const LWO3Chunk*> (child.get());
            BinaryReader reader (
                const_cast<char*> (reinterpret_cast<const char*> (chunk->getData().data())),
                static_cast<uint32_t> (chunk->getData().size()));
            surfaceName_ = reader.ReadNullTerminatedString();
            return;
        }
    }
}

void LWO3NodeGraph::extractNodeNames()
{
    // First find the NODS form within our surface
    const LWO3Form* nodsForm = nullptr;
    for (const auto& child : surfForm_->getChildren())
    {
        if (child->isForm() && child->getId() == LWO::NODS)
        {
            nodsForm = static_cast<const LWO3Form*> (child.get());
            break;
        }
    }

    if (!nodsForm)
    {
        // LOG (WARNING) << "No NODS form found in surface: " << surfaceName_;
        return;
    }

    // Now find all NSRV chunks within this NODS form's hierarchy
    LWO3Navigator nav;
    auto results = nav.findElementsById (nodsForm, LWO::NSRV);

    for (const auto& result : results)
    {
        const auto* nsrvChunk = static_cast<const LWO3Chunk*> (result.element);
        std::string nodeType = std::string (reinterpret_cast<const char*> (nsrvChunk->getData().data()));

        // Find associated NTAG form
        const LWO3Form* parent = static_cast<const LWO3Form*> (result.path.back());
        const LWO3Form* ntagForm = nullptr;

        bool foundNsrv = false;
        for (const auto& child : parent->getChildren())
        {
            if (foundNsrv && child->getId() == LWO::NTAG)
            {
                ntagForm = static_cast<const LWO3Form*> (child.get());
                break;
            }
            if (child.get() == nsrvChunk)
            {
                foundNsrv = true;
            }
        }

        if (!ntagForm)
        {
            // LOG (WARNING) << "Could not find NTAG form after NSRV in surface: " << surfaceName_;
            continue;
        }

        std::string refName;

        // Get node name from NNME chunk
        for (const auto& child : ntagForm->getChildren())
        {
            if (!child->isForm() && child->getId() == LWO::NNME)
            {
                const auto* chunk = static_cast<const LWO3Chunk*> (child.get());
                refName = std::string (reinterpret_cast<const char*> (chunk->getData().data()));
                break;
            }
        }

        if (!refName.empty())
        {
            nodeNames_.emplace_back (refName, nsrvChunk->getFileOffset());
            nodesByType_[nodeType].push_back (refName);
            // LOG (DBUG) << "Found node: " << refName << " of type: " << nodeType
                    //    << " in surface: " << surfaceName_;
        }
    }
}

std::vector<ImageNodeInfo> LWO3NodeGraph::getImageNodes() const
{
    std::vector<ImageNodeInfo> imageNodes;
    auto it = nodesByType_.find ("Image");
    if (it == nodesByType_.end())
    {
        return imageNodes;
    }

    LWO3Navigator nav;
    for (const std::string& nodeName : it->second)
    {
        ImageNodeInfo info;
        info.nodeName = nodeName;
        info.enabled = true;

        // Find this node's NTAG form
        auto nodeResults = nav.findElementsById (surfForm_, LWO::NTAG);
        for (const auto& result : nodeResults)
        {
            const auto* ntagForm = static_cast<const LWO3Form*> (result.element);

            // Verify this is our target node
            bool found = false;
            for (const auto& child : ntagForm->getChildren())
            {
                if (!child->isForm() && child->getId() == LWO::NNME)
                {
                    const auto* chunk = static_cast<const LWO3Chunk*> (child.get());
                    if (std::string (reinterpret_cast<const char*> (chunk->getData().data())) == nodeName)
                    {
                        found = true;
                        break;
                    }
                }
            }

            if (!found) continue;

            // Extract image data
            for (const auto& child : ntagForm->getChildren())
            {
                if (!child->isForm() && child->getId() == LWO::NSTA)
                {
                    const auto* chunk = static_cast<const LWO3Chunk*> (child.get());
                    if (!chunk->getData().empty())
                    {
                        info.enabled = chunk->getData()[0] != 0;
                    }
                }
                else if (child->isForm() && child->getId() == LWO::NDTA)
                {
                    const auto* ndtaForm = static_cast<const LWO3Form*> (child.get());
                    info.imagePath = extractImagePathFromNDTA (ndtaForm);
                    // LOG (DBUG) << "--------------------" <<  info.imagePath;
                    // Get UV map name
                    auto iuviResults = nav.findElementsById (ndtaForm, LWO::IUVI);
                    if (!iuviResults.empty())
                    {
                        const auto* iuviChunk = static_cast<const LWO3Chunk*> (iuviResults[0].element);
                        info.uvMapName = std::string (reinterpret_cast<const char*> (iuviChunk->getData().data()));
                    }

                    if (!info.imagePath.empty())
                    {
                        info.connections = getNodeConnections (nodeName);
                        // LOG (DBUG) << "Found image node: " << info.nodeName
                            //        << " using " << info.imagePath
                            //        << " in surface: " << surfaceName_;
                        imageNodes.push_back (info);
                        break;
                    }
                }
            }
        }
    }

    return imageNodes;
}

// Extract image path from NDTA form ensuring proper path formatting
std::string LWO3NodeGraph::extractImagePathFromNDTA (const LWO3Form* ndtaForm) const
{
    // Find IIMG form first
    LWO3Navigator nav;
    auto iimgResults = nav.findElementsById (ndtaForm, LWO::IIMG);
    if (!iimgResults.empty())
    {
        const auto* iimgForm = static_cast<const LWO3Form*> (iimgResults[0].element);

        // Find CLIP form
        auto clipResults = nav.findElementsById (iimgForm, LWO::CLIP);
        if (!clipResults.empty())
        {
            const auto* clipForm = static_cast<const LWO3Form*> (clipResults[0].element);

            // Find STIL form
            auto stilResults = nav.findElementsById (clipForm, LWO::STIL);
            if (!stilResults.empty())
            {
                const auto* stilForm = static_cast<const LWO3Form*> (stilResults[0].element);

                // Get path from anonymous chunk
                for (const auto& child : stilForm->getChildren())
                {
                    if (!child->isForm() &&
                        static_cast<const LWO3Chunk*> (child.get())->getId() == LWO::ANON)
                    {
                        const auto* chunk = static_cast<const LWO3Chunk*> (child.get());
                        std::string path = std::string (reinterpret_cast<const char*> (chunk->getData().data()));

                        // Fix path formatting - add slash after drive letter if missing
                        if (path.length() >= 2 && path[1] == ':' &&
                            (path.length() == 2 || (path[2] != '/' && path[2] != '\\')))
                        {
                            path.insert (2, 1, '/');
                            // LOG (DBUG) << "Fixed path formatting: " << path;
                        }

                        return path;
                    }
                }
            }
        }
    }
    return "";
}

std::vector<StandardNodeInfo> LWO3NodeGraph::getStandardNodes() const
{
    std::vector<StandardNodeInfo> standardNodes;
    auto it = nodesByType_.find ("Standard");
    if (it == nodesByType_.end())
    {
        return standardNodes;
    }

    LWO3Navigator nav;
    for (const std::string& nodeName : it->second)
    {
        StandardNodeInfo info;
        info.nodeName = nodeName;
        info.enabled = true; // Default to enabled

        // Find this node's NTAG form within surfForm_
        auto nodeResults = nav.findElementsById (surfForm_, LWO::NTAG);
        for (const auto& result : nodeResults)
        {
            const auto* ntagForm = static_cast<const LWO3Form*> (result.element);

            // Verify this is our target node by matching NNME
            bool found = false;
            for (const auto& child : ntagForm->getChildren())
            {
                if (!child->isForm() && child->getId() == LWO::NNME)
                {
                    const auto* chunk = static_cast<const LWO3Chunk*> (child.get());
                    if (std::string (reinterpret_cast<const char*> (chunk->getData().data())) == nodeName)
                    {
                        found = true;
                        break;
                    }
                }
            }

            if (!found) continue;

            // Process node data
            for (const auto& child : ntagForm->getChildren())
            {
                if (!child->isForm())
                {
                    if (child->getId() == LWO::NSTA)
                    {
                        const auto* chunk = static_cast<const LWO3Chunk*> (child.get());
                        if (!chunk->getData().empty())
                        {
                            info.enabled = chunk->getData()[0] != 0;
                        }
                    }
                }
                else if (child->getId() == LWO::NDTA)
                {
                    extractStandardValues (static_cast<const LWO3Form*> (child.get()), info);
                }
            }

            info.connections = getNodeConnections (nodeName);
            // LOG (DBUG) << "Found Standard node: " << info.nodeName
                     //   << " in surface: " << surfaceName_
                    //    << " enabled: " << info.enabled;

            standardNodes.push_back (info);
            break; // Found and processed our node
        }
    }

    return standardNodes;
}
std::vector<PrincipledBSDFInfo> LWO3NodeGraph::getPrincipledBSDFNodes() const
{
    std::vector<PrincipledBSDFInfo> bsdfNodes;
    auto it = nodesByType_.find ("Principled BSDF");
    if (it == nodesByType_.end())
    {
        return bsdfNodes;
    }

    LWO3Navigator nav;
    for (const std::string& nodeName : it->second)
    {
        PrincipledBSDFInfo info;
        info.nodeName = nodeName;
        info.enabled = true; // Default to enabled

        // Find this node's NTAG form
        auto nodeResults = nav.findElementsById (surfForm_, LWO::NTAG);
        for (const auto& result : nodeResults)
        {
            const auto* ntagForm = static_cast<const LWO3Form*> (result.element);

            // Verify this is our target node
            bool found = false;
            for (const auto& child : ntagForm->getChildren())
            {
                if (!child->isForm() && child->getId() == LWO::NNME)
                {
                    const auto* chunk = static_cast<const LWO3Chunk*> (child.get());
                    if (std::string (reinterpret_cast<const char*> (chunk->getData().data())) == nodeName)
                    {
                        found = true;
                        break;
                    }
                }
            }

            if (!found) continue;

            // Process node data
            for (const auto& child : ntagForm->getChildren())
            {
                if (!child->isForm())
                {
                    if (child->getId() == LWO::NSTA)
                    {
                        const auto* chunk = static_cast<const LWO3Chunk*> (child.get());
                        if (!chunk->getData().empty())
                        {
                            info.enabled = chunk->getData()[0] != 0;
                        }
                    }
                }
                else if (child->getId() == LWO::NDTA)
                {
                    extractPrincipledBSDFValues (static_cast<const LWO3Form*> (child.get()), info);
                }
            }

            info.connections = getNodeConnections (nodeName);
            // LOG (DBUG) << "Found Principled BSDF node: " << info.nodeName
                    //    << " in surface: " << surfaceName_
                      //  << " enabled: " << info.enabled;

            bsdfNodes.push_back (info);
            break; // Found and processed our node
        }
    }

    return bsdfNodes;
}

void LWO3NodeGraph::extractPrincipledBSDFValues (const LWO3Form* ndtaForm, PrincipledBSDFInfo& info) const
{
    // Helper lambda to find inner VALU form
    auto findInnerValu = [] (const LWO3Form* valuForm) -> const LWO3Form*
    {
        for (const auto& child : valuForm->getChildren())
        {
            if (child->isForm() && child->getId() == LWO::VALU)
            {
                return static_cast<const LWO3Form*> (child.get());
            }
        }
        return nullptr;
    };

    // Helper lambda to extract float value with proper nesting
    auto extractFloat = [&findInnerValu] (const LWO3Form* valuForm) -> float
    {
        const LWO3Form* innerValuForm = findInnerValu (valuForm);
        if (!innerValuForm) return 0.0f;

        for (const auto& child : innerValuForm->getChildren())
        {
            if (!child->isForm() &&
                static_cast<const LWO3Chunk*> (child.get())->getId() == LWO::ANON)
            {
                const auto* chunk = static_cast<const LWO3Chunk*> (child.get());
                if (chunk->getData().size() >= 12)
                { // Count (4) + double value (8)
                    BinaryReader reader (
                        const_cast<char*> (reinterpret_cast<const char*> (chunk->getData().data())),
                        static_cast<uint32_t> (chunk->getData().size()));

                    uint32_t count = mace::swap32 (reader.ReadUint32());
                    if (count == 1)
                    {
                        return static_cast<float> (mace::swapDouble (reader.ReadDouble()));
                    }
                }
            }
        }
        return 0.0f;
    };

    // Helper lambda to extract vector value with proper nesting
    auto extractVector3f = [&findInnerValu] (const LWO3Form* valuForm) -> Vector3f
    {
        const LWO3Form* innerValuForm = findInnerValu (valuForm);
        if (!innerValuForm) return Vector3f::Zero();

        for (const auto& child : innerValuForm->getChildren())
        {
            if (!child->isForm() &&
                static_cast<const LWO3Chunk*> (child.get())->getId() == LWO::ANON)
            {
                const auto* chunk = static_cast<const LWO3Chunk*> (child.get());
                if (chunk->getData().size() >= 28)
                { // Count (4) + 3 doubles (24)
                    BinaryReader reader (
                        const_cast<char*> (reinterpret_cast<const char*> (chunk->getData().data())),
                        static_cast<uint32_t> (chunk->getData().size()));

                    uint32_t count = mace::swap32 (reader.ReadUint32());
                    if (count == 3)
                    {
                        return Vector3f (
                            static_cast<float> (mace::swapDouble (reader.ReadDouble())),
                            static_cast<float> (mace::swapDouble (reader.ReadDouble())),
                            static_cast<float> (mace::swapDouble (reader.ReadDouble())));
                    }
                }
            }
        }
        return Vector3f::Zero();
    };

    // Find ATTR -> META -> ADAT path
    for (const auto& child : ndtaForm->getChildren())
    {
        if (child->isForm() && child->getId() == LWO::ATTR)
        {
            const auto* attrForm = static_cast<const LWO3Form*> (child.get());

            for (const auto& attrChild : attrForm->getChildren())
            {
                if (attrChild->isForm() && attrChild->getId() == LWO::META)
                {
                    const auto* metaForm = static_cast<const LWO3Form*> (attrChild.get());

                    for (const auto& metaChild : metaForm->getChildren())
                    {
                        if (metaChild->isForm() && metaChild->getId() == LWO::ADAT)
                        {
                            const auto* adatForm = static_cast<const LWO3Form*> (metaChild.get());

                            // Process each ENTR form for material properties
                            for (const auto& entrChild : adatForm->getChildren())
                            {
                                if (!entrChild->isForm() || entrChild->getId() != LWO::ENTR) continue;

                                const auto* entrForm = static_cast<const LWO3Form*> (entrChild.get());
                                std::string propName;
                                const LWO3Form* valuForm = nullptr;

                                // Get property name and VALU form
                                for (const auto& entry : entrForm->getChildren())
                                {
                                    if (!entry->isForm())
                                    {
                                        if (entry->getId() == LWO::NAME)
                                        {
                                            const auto* chunk = static_cast<const LWO3Chunk*> (entry.get());
                                            propName = std::string (reinterpret_cast<const char*> (chunk->getData().data()));
                                        }
                                    }
                                    else if (entry->getId() == LWO::VALU)
                                    {
                                        valuForm = static_cast<const LWO3Form*> (entry.get());
                                    }
                                }

                                if (!valuForm) continue;

                                // Map property names to PrincipledBSDFInfo fields
                                if (propName == "Color")
                                {
                                    info.baseColor = extractVector3f (valuForm);

                                    // for some reason R and B components are flipped so we have to swap
                                    std::swap (info.baseColor[0], info.baseColor[2]);

                                  //  mace::vecStr3f (info.baseColor, DBUG, "Base color");
                                }
                                else if (propName == "Luminous Color")
                                {
                                    info.luminousColor = extractVector3f (valuForm);
                                   // mace::vecStr3f (info.luminousColor, DBUG, "Luminous Color");
                                }
                                else if (propName == "Subsurface Color")
                                {
                                    info.subsurfaceColor = extractVector3f (valuForm);
                                   // mace::vecStr3f (info.subsurfaceColor, DBUG, "Subsurface Color");
                                }
                                else if (propName == "Transmittance")
                                {
                                    info.transmittance = extractVector3f (valuForm);
                                   // mace::vecStr3f (info.subsurfaceColor, DBUG, "Subsurface Color");
                                }
                                else
                                {
                                    float value = extractFloat (valuForm);
                                    if (propName == "Roughness")
                                        info.roughness = value;
                                    else if (propName == "Metallic")
                                        info.metallic = value;
                                    else if (propName == "Specular")
                                        info.specular = value;
                                    else if (propName == "Specular Tint")
                                        info.specularTint = value;
                                    else if (propName == "Sheen")
                                        info.sheen = value;
                                    else if (propName == "Sheen Tint")
                                        info.sheenTint = value;
                                    else if (propName == "Clearcoat")
                                        info.clearcoat = value;
                                    else if (propName == "Clearcoat Gloss")
                                        info.clearcoatGloss = value;
                                   // else if (propName == "Transmission")
                                    //    info.transmission = value;
                                    else if (propName == "Transparency")
                                        info.transparency = value;
                                    else if (propName == "Translucency")
                                        info.subsurface = value; // Maps to subsurface in the PBR model
                                    else if (propName == "Flatness")
                                        info.flatness = value;
                                    else if (propName == "Distance")
                                        info.subsurfaceDistance = value;
                                  /*  else if (propName == "Transmittance")
                                        info.transmittance = value;*/
                                    else if (propName == "Transmittance Distance")
                                        info.transmittanceDistance = value;
                                    else if (propName == "Anisotropic")
                                        info.anisotropic = value;
                                    else if (propName == "Rotation")
                                        info.anisotropicRotation = value;
                                    else if (propName == "Asymmetry")
                                        info.asymmetry = value;
                                    else if (propName == "Luminous")
                                        info.luminous = value;
                                    else if (propName == "Thin")
                                        info.thinWalled = value > 0.0f;
                                    else if (propName == "Refraction Index")
                                        info.ior = value;

                                    // LOG (DBUG) << "BSDF property " << propName << ": " << value;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

std::vector<NodeConnection> LWO3NodeGraph::getNodeConnections (const std::string& nodeName) const
{
    std::vector<NodeConnection> connections;

    // First find the NODS form
    const LWO3Form* nodsForm = nullptr;
    for (const auto& child : surfForm_->getChildren())
    {
        if (child->isForm() && child->getId() == LWO::NODS)
        {
            nodsForm = static_cast<const LWO3Form*> (child.get());
            break;
        }
    }

    if (!nodsForm)
    {
        return connections;
    }

    // Find NCON form in NODS
    LWO3Navigator nav;
    auto nconResults = nav.findElementsById (nodsForm, LWO::NCON);
    if (nconResults.empty())
    {
        return connections;
    }

    const auto* nconForm = static_cast<const LWO3Form*> (nconResults[0].element);

    // Process each connection in sequence
    NodeConnection currentConnection;
    bool processingConnection = false;

    for (const auto& child : nconForm->getChildren())
    {
        if (child->isForm()) continue;

        const auto* chunk = static_cast<const LWO3Chunk*> (child.get());
        std::string_view data (reinterpret_cast<const char*> (chunk->getData().data()));

        switch (chunk->getId())
        {
            case LWO::INME: // Target node name
                currentConnection = NodeConnection();
                processingConnection = true;
                currentConnection.targetNode = std::string (data);
                break;

            case LWO::IINM: // Input name
                if (processingConnection)
                {
                    currentConnection.inputName = std::string (data);
                }
                break;

            case LWO::IINN: // Source node name
                if (processingConnection)
                {
                    currentConnection.sourceNode = std::string (data);
                }
                break;

            case LWO::IONM: // Output name
                if (processingConnection)
                {
                    currentConnection.outputName = std::string (data);

                    // Only add the connection if it involves our target node
                    if (!currentConnection.sourceNode.empty() &&
                        !currentConnection.targetNode.empty() &&
                        !currentConnection.inputName.empty() &&
                        !currentConnection.outputName.empty() &&
                        (currentConnection.sourceNode == nodeName ||
                         currentConnection.targetNode == nodeName))
                    {
                        // Check for duplicates
                        bool isDuplicate = false;
                        for (const auto& existing : connections)
                        {
                            if (existing.sourceNode == currentConnection.sourceNode &&
                                existing.targetNode == currentConnection.targetNode &&
                                existing.inputName == currentConnection.inputName &&
                                existing.outputName == currentConnection.outputName)
                            {
                                isDuplicate = true;
                                break;
                            }
                        }

                        if (!isDuplicate)
                        {
                            // LOG (DBUG) << "Found connection: " << currentConnection.sourceNode
                                     //   << "." << currentConnection.outputName << " -> "
                                    //    << currentConnection.targetNode << "."
                                     //   << currentConnection.inputName;
                            connections.push_back (currentConnection);
                        }
                    }

                    processingConnection = false;
                }
                break;
        }
    }

    return connections;
}

std::vector<ImageConnection> LWO3NodeGraph::getConnectedImageNodes() const
{
    std::vector<ImageConnection> connections;
    auto images = getImageNodes();

    // For each image node
    for (const auto& img : images)
    {
        // Look at each connection from this image
        for (const auto& conn : img.connections)
        {
            // Check if this connects to a Principled BSDF node
            if (conn.targetNode.find ("Principled BSDF") != std::string::npos)
            {
                ImageConnection imgConn;
                imgConn.imageNode = img;
                imgConn.bsdfInputSocket = conn.inputName;
                connections.push_back (imgConn);

                // LOG (DBUG) << "Found image connection: " << img.nodeName
                          //  << " -> " << conn.targetNode
                          //  << " (" << conn.inputName << ")";
            }
        }
    }

    return connections;
}

bool LWO3NodeGraph::hasNode (const std::string& nodeName) const
{
    // Search nodeNames_ vector for a pair with matching node name
    return std::any_of (nodeNames_.begin(), nodeNames_.end(),
                        [&nodeName] (const auto& pair)
                        { return pair.first == nodeName; });
}

std::vector<NodeConnection> LWO3NodeGraph::processNodeConnections (const LWO3Form* nconForm,
                                                                   const std::string& nodeName) const
{
    std::vector<NodeConnection> connections;
    NodeConnection currentConnection;
    bool processingConnection = false;

    for (const auto& child : nconForm->getChildren())
    {
        if (child->isForm()) continue;

        const auto* chunk = static_cast<const LWO3Chunk*> (child.get());
        const auto& data = chunk->getData();

        switch (chunk->getId())
        {
            case LWO::INME: // Target node
                currentConnection = NodeConnection();
                processingConnection = true;
                currentConnection.targetNode = std::string (reinterpret_cast<const char*> (data.data()));
                break;

            case LWO::IINM: // Input name
                if (processingConnection)
                {
                    currentConnection.inputName = std::string (reinterpret_cast<const char*> (data.data()));
                }
                break;

            case LWO::IINN: // Source node name
                if (processingConnection)
                {
                    currentConnection.sourceNode = std::string (reinterpret_cast<const char*> (data.data()));
                }
                break;

            case LWO::IONM: // Output name
                if (processingConnection)
                {
                    currentConnection.outputName = std::string (reinterpret_cast<const char*> (data.data()));

                    // Add connection if:
                    // 1. All fields are populated
                    // 2. The connection involves our target node
                    // 3. Source and target are different nodes
                    if (!currentConnection.sourceNode.empty() &&
                        !currentConnection.targetNode.empty() &&
                        !currentConnection.inputName.empty() &&
                        !currentConnection.outputName.empty() &&
                        currentConnection.sourceNode != currentConnection.targetNode &&
                        (currentConnection.sourceNode == nodeName ||
                         currentConnection.targetNode == nodeName))
                    {
                        // Check for duplicates
                        bool isDuplicate = false;
                        for (const auto& existing : connections)
                        {
                            if (existing.sourceNode == currentConnection.sourceNode &&
                                existing.targetNode == currentConnection.targetNode &&
                                existing.inputName == currentConnection.inputName &&
                                existing.outputName == currentConnection.outputName)
                            {
                                isDuplicate = true;
                                break;
                            }
                        }

                        if (!isDuplicate)
                        {
                            // LOG (DBUG) << "Adding connection: " << currentConnection.sourceNode
                                //       << "." << currentConnection.outputName << " -> "
                                //       << currentConnection.targetNode << "."
                                 //      << currentConnection.inputName;
                            connections.push_back (currentConnection);
                        }
                    }

                    processingConnection = false;
                }
                break;
        }
    }

    return connections;
}

void LWO3NodeGraph::extractStandardValues (const LWO3Form* ndtaForm, StandardNodeInfo& info) const
{
    // Find SATR -> META -> ADAT path
    for (const auto& child : ndtaForm->getChildren())
    {
        if (child->isForm() && child->getId() == LWO::SATR)
        {
            const auto* satrForm = static_cast<const LWO3Form*> (child.get());

            for (const auto& satrChild : satrForm->getChildren())
            {
                if (satrChild->isForm() && satrChild->getId() == LWO::META)
                {
                    const auto* metaForm = static_cast<const LWO3Form*> (satrChild.get());

                    for (const auto& metaChild : metaForm->getChildren())
                    {
                        if (metaChild->isForm() && metaChild->getId() == LWO::ADAT)
                        {
                            const auto* adatForm = static_cast<const LWO3Form*> (metaChild.get());

                            // Helper lambda to find inner VALU form
                            auto findInnerValu = [] (const LWO3Form* valuForm) -> const LWO3Form*
                            {
                                for (const auto& child : valuForm->getChildren())
                                {
                                    if (child->isForm() && child->getId() == LWO::VALU)
                                    {
                                        return static_cast<const LWO3Form*> (child.get());
                                    }
                                }
                                return nullptr;
                            };

                            // Helper lambda to extract float value from inner VALU form
                            auto extractFloat = [&findInnerValu] (const LWO3Form* valuForm) -> float
                            {
                                const LWO3Form* innerValuForm = findInnerValu (valuForm);
                                if (!innerValuForm) return 0.0f;

                                for (const auto& child : innerValuForm->getChildren())
                                {
                                    if (!child->isForm() &&
                                        static_cast<const LWO3Chunk*> (child.get())->getId() == LWO::ANON)
                                    {
                                        const auto* chunk = static_cast<const LWO3Chunk*> (child.get());
                                        if (chunk->getData().size() >= 12)
                                        { // Count(4) + double(8)
                                            BinaryReader reader (
                                                const_cast<char*> (reinterpret_cast<const char*> (chunk->getData().data())),
                                                static_cast<uint32_t> (chunk->getData().size()));

                                            uint32_t count = mace::swap32 (reader.ReadUint32());
                                            if (count == 1)
                                            {
                                                return static_cast<float> (mace::swapDouble (reader.ReadDouble()));
                                            }
                                        }
                                    }
                                }
                                return 0.0f;
                            };

                            // Helper lambda to extract Vector3f value from inner VALU form
                            auto extractVector3f = [&findInnerValu] (const LWO3Form* valuForm) -> Vector3f
                            {
                                const LWO3Form* innerValuForm = findInnerValu (valuForm);
                                if (!innerValuForm) return Vector3f::Zero();

                                for (const auto& child : innerValuForm->getChildren())
                                {
                                    if (!child->isForm() &&
                                        static_cast<const LWO3Chunk*> (child.get())->getId() == LWO::ANON)
                                    {
                                        const auto* chunk = static_cast<const LWO3Chunk*> (child.get());
                                        if (chunk->getData().size() >= 28)
                                        { // Count(4) + 3*double(24)
                                            BinaryReader reader (
                                                const_cast<char*> (reinterpret_cast<const char*> (chunk->getData().data())),
                                                static_cast<uint32_t> (chunk->getData().size()));

                                            uint32_t count = mace::swap32 (reader.ReadUint32());
                                            if (count == 3)
                                            {
                                               return Vector3f (
                                                    static_cast<float> (mace::swapDouble (reader.ReadDouble())),
                                                    static_cast<float> (mace::swapDouble (reader.ReadDouble())),
                                                    static_cast<float> (mace::swapDouble (reader.ReadDouble())));
                                              
                                            }
                                        }
                                    }
                                }
                                return Vector3f::Zero();
                            };

                            // Process each ENTR form for material properties
                            for (const auto& entrChild : adatForm->getChildren())
                            {
                                if (!entrChild->isForm() || entrChild->getId() != LWO::ENTR) continue;

                                const auto* entrForm = static_cast<const LWO3Form*> (entrChild.get());
                                std::string propName;
                                const LWO3Form* valuForm = nullptr;

                                // Get property name and VALU form
                                for (const auto& entry : entrForm->getChildren())
                                {
                                    if (!entry->isForm())
                                    {
                                        if (entry->getId() == LWO::NAME)
                                        {
                                            const auto* chunk = static_cast<const LWO3Chunk*> (entry.get());
                                            propName = std::string (reinterpret_cast<const char*> (chunk->getData().data()));
                                        }
                                    }
                                    else if (entry->getId() == LWO::VALU)
                                    {
                                        valuForm = static_cast<const LWO3Form*> (entry.get());
                                    }
                                }

                                if (!valuForm) continue;

                                // Map property names to StandardNodeInfo fields
                                if (propName == "Color")
                                {
                                    Vector3f c = extractVector3f (valuForm);

                                    // NB.  for some unknown to me reason, it appears that
                                    // LW is saving color in BGR instead of RGB
                                    // Swap x and z components
                                    std::swap (c.x(),c.z());
                                    
                                    info.color = c;
                                    // LOG (DBUG) << info.color.x() << ", " << info.color.y() << ", " << info.color.z();
                                    // LOG (DBUG) << "Standard Color: " << info.color.transpose();

                                    
                                }
                                else
                                {
                                    float value = extractFloat (valuForm);
                                    if (propName == "Luminosity")
                                        info.luminosity = value;
                                    else if (propName == "Diffuse")
                                        info.diffuse = value;
                                    else if (propName == "Specular")
                                        info.specular = value;
                                    else if (propName == "Glossiness")
                                        info.glossiness = value;
                                    else if (propName == "Reflection")
                                        info.reflection = value;
                                    else if (propName == "Transparency")
                                        info.transparency = value;
                                    else if (propName == "Refraction Index")
                                        info.refractionIndex = value;
                                    else if (propName == "Translucency")
                                        info.translucency = value;
                                    else if (propName == "Color Highlight")
                                        info.colorHighlight = value;
                                    else if (propName == "Color Filter")
                                        info.colorFilter = value;
                                    else if (propName == "Diffuse Sharpness")
                                        info.diffuseSharpness = value;
                                    else if (propName == "Bump Height")
                                        info.bumpHeight = value;

                                    // LOG (DBUG) << "Standard property " << propName << ": " << value;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}