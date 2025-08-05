

LWO3ToCgModelConverter::LWO3ToCgModelConverter (ConversionFlags flags) :
    flags_ (flags)
{
}

CgModelPtr LWO3ToCgModelConverter::convert (const LWO3Layer* layer)
{
    if (!validateLayer (layer))
    {
        return nullptr;
    }

    auto model = CgModel::create();
    model->contentDirectory = contentDir_;

    // Clear tracking state at start of conversion
    processedImages_.clear();
    processedSurfaces_.clear();
    imageTextureMap_.clear();

    // Convert geometry
    if ((flags_ & ConversionFlags::StandardGeometry) != ConversionFlags::None)
    {
        if (!convertVertices (layer, model))
        {
            LOG (WARNING) << "Failed to convert vertices: " << errorMsg_;
            return nullptr;
        }

        if (!convertTriangles (layer, model))
        {
            LOG (WARNING) << "Failed to convert triangles: " << errorMsg_;
            return nullptr;
        }

        if (!convertUVs (layer, model))
        {
            LOG (WARNING) << "Failed to convert UVs: " << errorMsg_;
            return nullptr;
        }
    }

    // Convert materials (which includes images and textures)
    if ((flags_ & ConversionFlags::Materials) == ConversionFlags::Materials)
    {
        // Get initial image count for allocation
        auto imageNodes = layer->getSurfaces()[0]->getNodeGraph()->getImageNodes();
        model->images.reserve (imageNodes.size());
        model->textures.reserve (imageNodes.size());

        if (!convertImages (layer, model))
        {
            LOG (WARNING) << "Failed to convert images: " << errorMsg_;
            return nullptr;
        }

        if (!convertMaterials (layer, model))
        {
            LOG (WARNING) << "Failed to convert materials: " << errorMsg_;
            return nullptr;
        }
    }

    return model;
}

bool LWO3ToCgModelConverter::validateLayer (const LWO3Layer* layer)
{
    if (!layer)
    {
        errorMsg_ = "Null layer provided";
        return false;
    }

    // Check polygon type is FACE
    if (layer->getPolygonType() != sabi::LWO::FACE)
    {
        errorMsg_ = "Only FACE polygon type is supported";
        return false;
    }

    // Basic geometry checks
    if (layer->getPoints().empty())
    {
        errorMsg_ = "Layer contains no vertices";
        return false;
    }

    if (layer->getPolygons().empty())
    {
        errorMsg_ = "Layer contains no polygons";
        return false;
    }

    // Verify all polygons are triangles
    for (const auto& poly : layer->getPolygons())
    {
        if (poly.indices.size() != 3)
        {
            errorMsg_ = "Non-triangle polygon found";
            return false;
        }
    }

    return true;
}

bool LWO3ToCgModelConverter::convertVertices (const LWO3Layer* layer, CgModelPtr model)
{
    const auto& points = layer->getPoints();

    // Resize vertex matrix
    model->V.resize (3, points.size());

    // Copy vertices to matrix
    for (size_t i = 0; i < points.size(); ++i)
    {
        model->V.col (i) = points[i];
    }

    return true;
}

bool LWO3ToCgModelConverter::convertTriangles (const LWO3Layer* layer, CgModelPtr model)
{
    const auto& polygons = layer->getPolygons();

    // Create a single surface for all triangles initially
    CgModelSurface surface;
    surface.F.resize (3, polygons.size());

    // Copy triangle indices
    for (size_t i = 0; i < polygons.size(); ++i)
    {
        const auto& poly = polygons[i];
        surface.F.col (i) = Vector3u (
            poly.indices[0],
            poly.indices[1],
            poly.indices[2]);
    }

    // Add surface to model
    model->S.push_back (surface);
    model->triCount = polygons.size();

    return true;
}

bool LWO3ToCgModelConverter::convertUVs (const LWO3Layer* layer, CgModelPtr model)
{

    auto uvMaps = layer->getVertexMaps (sabi::LWO::TXUV);
    if (uvMaps.empty())
    {
        errorMsg_ = "No UV maps found";
        return false;
    }

    const VertexMap* primaryUVMap = uvMaps[0];
    if (!primaryUVMap || primaryUVMap->dimension != 2)
    {
        errorMsg_ = "Invalid UV map dimension";
        return false;
    }

    // Initialize UV matrix
    model->UV0.resize (2, model->V.cols());
    model->UV0.setZero();

    // Copy UV data - values are already properly parsed
    for (size_t i = 0; i < primaryUVMap->vertexIndices.size(); ++i)
    {
        uint32_t vertIndex = primaryUVMap->vertexIndices[i];
        if (vertIndex >= static_cast<uint32_t> (model->V.cols()))
        {
            LOG (WARNING) << "UV vertex index out of bounds: " << vertIndex;
            continue;
        }

        model->UV0.col (vertIndex) = Eigen::Vector2f (
            primaryUVMap->values[i * 2],    // U coordinate
            primaryUVMap->values[i * 2 + 1] // V coordinate
        );
    }

    // Handle UV1 similarly if present
    if (uvMaps.size() > 1)
    {
        const VertexMap* secondaryUVMap = uvMaps[1];
        if (secondaryUVMap && secondaryUVMap->dimension == 2)
        {
            model->UV1.resize (2, model->V.cols());
            model->UV1.setZero();

            for (size_t i = 0; i < secondaryUVMap->vertexIndices.size(); ++i)
            {
                uint32_t vertIndex = secondaryUVMap->vertexIndices[i];
                if (vertIndex >= static_cast<uint32_t> (model->V.cols()))
                {
                    continue;
                }

                model->UV1.col (vertIndex) = Eigen::Vector2f (
                    secondaryUVMap->values[i * 2],
                    secondaryUVMap->values[i * 2 + 1]);
            }
        }
    }

    return true;
}

// Pre-allocate vectors to avoid reallocation:
bool LWO3ToCgModelConverter::convertImages (const LWO3Layer* layer, CgModelPtr model)
{
    auto imageNodes = layer->getSurfaces()[0]->getNodeGraph()->getImageNodes();
    model->images.reserve (imageNodes.size());
    model->textures.reserve (imageNodes.size());
    processedImages_.reserve (imageNodes.size());

    // Process each surface's node graph
    for (const auto& surface : layer->getSurfaces())
    {
        if (!surface || !surface->getNodeGraph())
        {
            continue;
        }

        const auto* nodeGraph = surface->getNodeGraph();
        auto connectedImages = nodeGraph->getConnectedImageNodes();

        for (const auto& conn : connectedImages)
        {
            const auto& imageNode = conn.imageNode;
            if (imageNode.imagePath.empty() || !imageNode.enabled)
            {
                continue;
            }

            // Add image if not already processed
            if (processedImages_.find (imageNode.imagePath) == processedImages_.end())
            {
                Image image;
                image.uri = imageNode.imagePath;

                if (!contentDir_.empty())
                {
                    fs::path fullPath = contentDir_ / image.uri;
                    if (fs::exists (fullPath))
                    {
                        image.uri = fullPath.string();
                    }
                }

                model->images.push_back (image);
                size_t imageIndex = model->images.size() - 1;
                processedImages_[imageNode.imagePath] = imageIndex;

                // Create corresponding texture
                Texture texture;
                texture.name = imageNode.nodeName;
                texture.source = static_cast<int> (imageIndex);
                model->textures.push_back (texture);

                // Store mapping between image path and texture index
                imageTextureMap_[imageNode.imagePath] = model->textures.size() - 1;
            }
        }
    }

    return true;
}
bool LWO3ToCgModelConverter::convertMaterials (const LWO3Layer* layer, CgModelPtr model)
{
    if (!model->S.empty())
    { // Model must have a surface
        for (const auto& surface : layer->getSurfaces())
        {
            if (!surface || !surface->getNodeGraph())
            {
                continue;
            }

            if (processedSurfaces_.find (surface->getName()) != processedSurfaces_.end())
            {
                continue;
            }
            processedSurfaces_.insert (surface->getName());

            auto bsdfNodes = surface->getNodeGraph()->getPrincipledBSDFNodes();
            if (bsdfNodes.empty())
            {
                LOG (WARNING) << "No BSDF nodes found for surface: " << surface->getName();
                continue; // Skip this surface but continue processing others
            }

            Material material;
            material.name = surface->getName();
            const auto& bsdf = bsdfNodes[0];
            material.pbrMetallicRoughness.baseColorFactor = {
                bsdf.baseColor (0), bsdf.baseColor (1), bsdf.baseColor (2), 1.0f};
            material.pbrMetallicRoughness.metallicFactor = bsdf.metallic;
            material.pbrMetallicRoughness.roughnessFactor = bsdf.roughness;

            auto imageNodes = surface->getNodeGraph()->getConnectedImageNodes();

            for (const auto& conn : imageNodes)
            {
                const auto& imageNode = conn.imageNode;
                if (imageNode.imagePath.empty() || !imageNode.enabled)
                {
                    continue;
                }

                // Use existing texture index from convertImages
                auto it = imageTextureMap_.find (imageNode.imagePath);
                if (it == imageTextureMap_.end())
                {
                    LOG (WARNING) << "Missing texture for image: " << imageNode.imagePath;
                    continue;
                }

                // Create texture info using existing texture index
                TextureInfo texInfo;
                texInfo.textureIndex = static_cast<int> (it->second);
                texInfo.texCoord = 0;

                if (conn.bsdfInputSocket == "Color")
                {
                    material.pbrMetallicRoughness.baseColorTexture = texInfo;
                }
                // Handle other texture types...
            }
            model->S[0].material = material;
        }
        return true;
    }

    errorMsg_ = "No surfaces present in model";
    return false;
}

#if 0
bool LWO3ToCgModelConverter::convertMaterials (const LWO3Layer* layer, CgModelPtr model)
{
    if ((flags_ & ConversionFlags::Materials) != ConversionFlags::Materials)
    {
        return true;
    }

    for (const auto& surface : layer->getSurfaces())
    {
        if (!surface || !surface->getNodeGraph())
        {
            continue;
        }

        if (processedSurfaces_.find (surface->getName()) != processedSurfaces_.end())
        {
            continue;
        }
        processedSurfaces_.insert (surface->getName());

        auto bsdfNodes = surface->getNodeGraph()->getPrincipledBSDFNodes();
        if (bsdfNodes.empty())
        {
            continue;
        }

        Material material;
        material.name = surface->getName();

        const auto& bsdf = bsdfNodes[0];
        material.pbrMetallicRoughness.baseColorFactor = {
            bsdf.baseColor (0), bsdf.baseColor (1), bsdf.baseColor (2), 1.0f};
        material.pbrMetallicRoughness.metallicFactor = bsdf.metallic;
        material.pbrMetallicRoughness.roughnessFactor = bsdf.roughness;

        auto imageNodes = surface->getNodeGraph()->getConnectedImageNodes();

        for (const auto& conn : imageNodes)
        {
            const auto& imageNode = conn.imageNode;
            if (imageNode.imagePath.empty() || !imageNode.enabled)
            {
                continue;
            }

            // Use existing texture index from convertImages
            auto it = imageTextureMap_.find (imageNode.imagePath);
            if (it == imageTextureMap_.end())
            {
                LOG (WARNING) << "Missing texture for image: " << imageNode.imagePath;
                continue;
            }

            // Create texture info using existing texture index
            TextureInfo texInfo;
            texInfo.textureIndex = static_cast<int> (it->second);
            texInfo.texCoord = 0;

            if (conn.bsdfInputSocket == "Color")
            {
                material.pbrMetallicRoughness.baseColorTexture = texInfo;
            }
            // Handle other texture types...
        }
        if (!model->S.empty() && !bsdfNodes.empty())
        {
            model->S[0].material = material;
        }
       
    }

    return true;
}
#endif