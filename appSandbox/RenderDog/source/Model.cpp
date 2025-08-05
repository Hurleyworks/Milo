#include "Model.h"



Model::Model()
{
    imageCache = ImageCacheHandler::create();
    loadStrategy = std::make_shared<sabi::NormalizedClump>();
}

Model::~Model()
{
    framework.shutdown();
}


void Model::addNodeToRenderer (RenderableNode node)
{

    framework.render.getMessenger().send (QMS::addWeakNode (node));
   
}

void Model::loadHDRIfromIcon (const std::filesystem::path& iconPath)
{
    // the full hdr image should have the same name as the icon image
    std::string hdrName = iconPath.stem().string();
    std::string contentFolder = properties.renderProps->getVal<std::string> (RenderKey::ExternalContentFolder);
    std::filesystem::path hdrPath (contentFolder + "/HDRI/" + hdrName + ".hdr");
    if (!std::filesystem::exists (hdrPath))
        throw std::runtime_error ("file does not exist: " + hdrPath.string());

    addSkyDomeImage (hdrPath);
}

void Model::loadCgModelfromIcon (const std::filesystem::path& iconPath)
{
    std::string modelName = iconPath.stem().string();
    std::string contentFolder = properties.renderProps->getVal<std::string> (RenderKey::ExternalContentFolder);
    std::filesystem::path cgModelFolder (contentFolder + "/models/" + modelName);
    if (!std::filesystem::exists (cgModelFolder))
        throw std::runtime_error ("file does not exist: " + cgModelFolder.string());

    std::vector<std::string> modelFiles;
    modelFiles.push_back (cgModelFolder.generic_string());
    onDrop (modelFiles);
}

void Model::processPath (const std::filesystem::path& p)
{
    if (!std::filesystem::exists (p))
        throw std::runtime_error ("file does not exist: " + p.string());

    if (std::filesystem::is_directory (p))
    {
        for (const auto& entry : std::filesystem::directory_iterator (p))
        {
            processPath (entry.path());
        }
    }
    else
    {
        std::string fileExtension = p.extension().string();

        /// handle cgModels in different formats // just gltf for now
        if (isSupportedMeshFormat (fileExtension))
        {
            bool validPath = true;

            // Check if the extension is .gltf, .GLTF, .glb, or .GLB
            std::string ext = p.extension().string();
            if (ext != ".gltf" && ext != ".GLTF" && ext != ".glb" && ext != ".GLB")
            {
                validPath = false;
            }

            // Check for specific invalid path conditions
            if (!isValidPath (p, "Draco") ||
                !isValidPath (p, "KTX") ||
                !isValidPath (p, "JPG") ||
                !isValidPath (p, "Unicode"))
            {
                validPath = false;
            }

            // Add to modelPaths only if all conditions are met
            if (validPath)
            {
                modelPaths.push_back (p);
            }
        }
        else if (isSupportedImageFormat (fileExtension))
        {
            // Handle HDR/EXR files directly
            if (fileExtension == ".hdr" || fileExtension == ".HDR" || 
                fileExtension == ".exr" || fileExtension == ".EXR" ||
                fileExtension == ".hdri" || fileExtension == ".HDRI")
            {
                // Load HDR file directly
                addSkyDomeImage(p);
                LOG(DBUG) << "Loading HDR file: " << p.string();
            }
            else if (p.extension() == ".png" || p.extension() == ".PNG")
            {
                if (pathContainsIgnoreCase (p, "hdri_thumbs"))
                    hdrIconPaths.push_back (p);

                if (pathContainsIgnoreCase (p, "model_thumbs"))
                    modelIconPaths.push_back (p);
            }
            else if (p.extension() == ".jpg" || p.extension() == ".JPG")
            {
                if (pathContainsIgnoreCase (p, "model_thumbs"))
                    modelIconPaths.push_back (p);
            }
        }
    }
}

void Model::initialize (CameraHandle camera, const PropertyService& properties)
{
    this->properties = properties;
    this->camera = camera;

    framework.init (properties);

    // send image cache to Imaging
   // framework.imaging.getMessenger().send (QMS::initImageServices (imageCache));

    // send camera and image cache to the renderer
    framework.render.getMessenger().send (QMS::initRenderEngine (camera, imageCache));
}

void Model::onDrop (const std::vector<std::string>& filenames)
{
    // Commented out test code - was creating a mesh light instead of processing files
    // // Create a warm white mesh light with custom size and intensity
    //  warmLight = sabi::MeshOps::createLuminousRectangleNode (
    //      "WarmMeshLight",
    //      4.0f, 2.0f,                         // 4x2 units
    //      Eigen::Vector3f (1.0f, 0.9f, 0.8f), // Warm white color
    //      10.0f                               // High intensity
    // );

    //  warmLight->setClientID (warmLight->getID());
    //  sabi::SpaceTime& st = warmLight->getSpaceTime();
    //  st.worldTransform.translation() = Eigen::Vector3f (0.0f, 1.0f, -4.0f);
    //  addNodeToRenderer (warmLight);

    // /*  groundPlane = sabi::MeshOps::createGroundPlaneNode();
    //  groundPlane->setClientID (groundPlane->getID());

    //   addNodeToRenderer (groundPlane);*/

    //   return;

    try
    {
        modelPaths.clear();
        hdrIconPaths.clear();
        modelIconPaths.clear();

        for (const auto& filename : filenames)
        {
            std::filesystem::path p (filename);
            processPath (p);
        }

        if (modelPaths.size() == 1)
        {
            // Load single GLTF file
            loadGLTF(modelPaths[0]);
        }
        else if (modelPaths.size() > 1)
        {
            // Load multiple GLTF files
            for (const auto& modelPath : modelPaths)
            {
                loadGLTF(modelPath);
            }
        }

        if (hdrIconPaths.size())
        {
            // load hdr icons concurrently in Dreamer
            framework.dreamer.getMessenger().send (QMS::loadImageList (hdrIconPaths));
        }

        if (modelIconPaths.size())
        {
            // load model icons concurrently in Dreamer
            framework.dreamer.getMessenger().send (QMS::loadImageList (modelIconPaths));
        }
    }
    catch (std::exception& e)
    {
        LOG (CRITICAL) << e.what();
    }
}
