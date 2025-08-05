#include "CommandProcessor.h"
#include "View.h"

using nanogui::Vector4i;

CommandProcessor::CommandProcessor (View* gui, const PropertyService& properties) :
    gui (gui),
    properties (properties)
{
}

std::string CommandProcessor::processCommand (const std::string& cmd)
{
    LOG (DBUG) << cmd;

    // Handle ping command
    if (cmd == "ping")
    {
        return processPingCommand();
    }

    // Handle camera info command
    if (cmd == "GetCameraInfo")
    {
        return processCameraInfoCommand();
    }

    // Handle render settings
    if (cmd == "GetRenderSettings")
    {
        return processGetRenderSettingsCommand();
    }

    // Handle camera movement commands
    if (cmd.substr (0, 15) == "SetCameraLookAt")
    {
        return processCameraLookAtCommand (cmd);
    }

    if (cmd.substr (0, 10) == "CameraZoom")
    {
        return processCameraZoomCommand (cmd);
    }

    if (cmd.substr (0, 9) == "CameraPan")
    {
        return processCameraPanCommand (cmd);
    }

    // Handle render property commands
    if (cmd.substr (0, 15) == "SetRenderPasses")
    {
        return processSetRenderPassesCommand (cmd);
    }

    if (cmd.substr (0, 16) == "SetPreviewScale")
    {
        return processSetPreviewScaleCommand (cmd);
    }

    if (cmd.substr (0, 11) == "SetHDRImage")
    {
        return processSetHDRImageCommand (cmd);
    }

    if (cmd.substr (0, 8) == "LoadGLTF")
    {
        return processLoadGLTFCommand (cmd);
    }

    if (cmd.substr (0, 14) == "LoadGltfFolder")
    {
        return processLoadGltfFolderCommand (cmd);
    }

    if (cmd.substr (0, 19) == "SetContentDirectory")
    {
        return processSetContentDirectoryCommand (cmd);
    }

    // Handle background color command
    if (cmd.substr (0, 18) == "SetBackgroundColor")
    {
        return processBackgroundColorCommand (cmd);
    }

    // Handle cube rotation command
    if (cmd.substr (0, 15) == "SetCubeRotation")
    {
        return processCubeRotationCommand (cmd);
    }

    // Handle pipeline switching command
    if (cmd.substr (0, 11) == "SetPipeline")
    {
        return processSetPipelineCommand (cmd);
    }

    // Handle get available pipelines command
    if (cmd == "GetAvailablePipelines")
    {
        return processGetAvailablePipelinesCommand();
    }

    // Unknown command
    return "Unknown command: " + cmd;
}

std::string CommandProcessor::processGetRenderSettingsCommand()
{
    try
    {
        uint32_t renderPasses = properties.renderProps->getVal<uint32_t> (RenderKey::RenderPasses);
        PreviewScaleFactor previewScale = properties.renderProps->getVal<PreviewScaleFactor> (RenderKey::RenderScale);
        std::string contentFolder = properties.renderProps->getVal<std::string> (RenderKey::ContentFolder);

        std::ostringstream response;
        response << "{"
                 << "\"renderPasses\":" << renderPasses << ","
                 << "\"previewScale\":" << static_cast<int> (previewScale) << ","
                 << "\"contentDirectory\":\"" << contentFolder << "\""
                 << "}";

        return response.str();
    }
    catch (const std::exception& e)
    {
        return "Error getting render settings: " + std::string (e.what());
    }
}

std::string CommandProcessor::processLoadGLTFCommand (const std::string& cmd)
{
    // Parse command: "SetHDRImage path"
    std::istringstream iss (cmd);
    std::string command, gltfPath;

    if (!(iss >> command))
    {
        return "Error: Invalid LoadGLTF format. Expected: LoadGLTF path";
    }

    // Get the rest of the line as the path (may contain spaces)
    std::getline (iss, gltfPath);

    // Trim leading whitespace
    gltfPath.erase (0, gltfPath.find_first_not_of (" \t"));

    if (gltfPath.empty())
    {
        return "Error: GLTF file path cannot be empty";
    }

    try
    {
        std::filesystem::path gltfFilePath (gltfPath);

        // Check if file exists
        if (!std::filesystem::exists (gltfFilePath))
        {
            return "Error: HDR image file does not exist: " + gltfPath;
        }

        // Check if it's a supported HDR format
        std::string extension = gltfFilePath.extension().string();
        std::transform (extension.begin(), extension.end(), extension.begin(), ::tolower);

        if (extension != ".gltf" && extension != ".glb" && extension != ".GLTF" && extension != ".GLB")
        {
            return "Error: Unsupported Gltf format. Supported formats: .gltf, .glb, .GLTF, .GLB";
        }

        // Emit signal to notify Model of HDR image change
        loadGltfEmitter.fire (gltfFilePath);

        std::ostringstream response;
        response << "Gltf file set to: " << gltfPath;
        return response.str();
    }
    catch (const std::exception& e)
    {
        return "Error setting HDR image: " + std::string (e.what());
    }
}

std::string CommandProcessor::processLoadGltfFolderCommand (const std::string& cmd)
{
    // Parse command: "SetHDRImage path"
    std::istringstream iss (cmd);
    std::string command, gltfFolder;

    if (!(iss >> command))
    {
        return "Error: Invalid LoadGLTF format. Expected: LoadGLTF path";
    }

    // Get the rest of the line as the path (may contain spaces)
    std::getline (iss, gltfFolder);

    // Trim leading whitespace
    gltfFolder.erase (0, gltfFolder.find_first_not_of (" \t"));

    if (gltfFolder.empty())
    {
        return "Error: GLTF folder path cannot be empty";
    }

    try
    {
        std::filesystem::path gltfFolderPath (gltfFolder);

        // Check if file exists
        if (!std::filesystem::is_directory (gltfFolderPath))
        {
            return "Error: Gltf folder does not exist: " + gltfFolder;
        }

      

        // Emit signal to notify Model of HDR image change
        loadGltfFolderEmitter.fire (gltfFolderPath);

        std::ostringstream response;
        response << "Gltf folder set to: " << gltfFolder;
        return response.str();
    }
    catch (const std::exception& e)
    {
        return "Error setting HDR image: " + std::string (e.what());
    }
}

std::string CommandProcessor::processSetRenderPassesCommand (const std::string& cmd)
{
    // Parse command: "SetRenderPasses count"
    std::istringstream iss (cmd);
    std::string command;
    uint32_t passes;

    if (!(iss >> command >> passes))
    {
        return "Error: Invalid SetRenderPasses format. Expected: SetRenderPasses count";
    }

    // Validate passes count (reasonable range)
    if (passes < 1 || passes > 10)
    {
        return "Error: Render passes must be between 1 and 10";
    }

    try
    {
        properties.renderProps->setValue (RenderKey::RenderPasses, passes);

        std::ostringstream response;
        response << "Render passes set to " << passes;
        return response.str();
    }
    catch (const std::exception& e)
    {
        return "Error setting render passes: " + std::string (e.what());
    }
}

std::string CommandProcessor::processSetPreviewScaleCommand (const std::string& cmd)
{
    // Parse command: "SetPreviewScale factor"
    std::istringstream iss (cmd);
    std::string command;
    int scaleFactor;

    if (!(iss >> command >> scaleFactor))
    {
        return "Error: Invalid SetPreviewScale format. Expected: SetPreviewScale factor (1, 2, or 4)";
    }

    PreviewScaleFactor scale;
    switch (scaleFactor)
    {
        case 1:
            scale = PreviewScaleFactor::x1;
            break;
        case 2:
            scale = PreviewScaleFactor::x2;
            break;
        case 4:
            scale = PreviewScaleFactor::x4;
            break;
        default:
            return "Error: Invalid scale factor. Must be 1, 2, or 4";
    }

    try
    {
        properties.renderProps->setValue (RenderKey::RenderScale, scale);

        std::ostringstream response;
        response << "Preview scale factor set to x" << scaleFactor;
        return response.str();
    }
    catch (const std::exception& e)
    {
        return "Error setting preview scale: " + std::string (e.what());
    }
}

std::string CommandProcessor::processSetHDRImageCommand (const std::string& cmd)
{
    // Parse command: "SetHDRImage path"
    std::istringstream iss (cmd);
    std::string command, hdrPath;

    if (!(iss >> command))
    {
        return "Error: Invalid SetHDRImage format. Expected: SetHDRImage path";
    }

    // Get the rest of the line as the path (may contain spaces)
    std::getline (iss, hdrPath);

    // Trim leading whitespace
    hdrPath.erase (0, hdrPath.find_first_not_of (" \t"));

    if (hdrPath.empty())
    {
        return "Error: HDR image path cannot be empty";
    }

    try
    {
        std::filesystem::path hdrFilePath (hdrPath);

        // Check if file exists
        if (!std::filesystem::exists (hdrFilePath))
        {
            return "Error: HDR image file does not exist: " + hdrPath;
        }

        // Check if it's a supported HDR format
        std::string extension = hdrFilePath.extension().string();
        std::transform (extension.begin(), extension.end(), extension.begin(), ::tolower);

        if (extension != ".hdr" && extension != ".exr" && extension != ".hdri")
        {
            return "Error: Unsupported HDR format. Supported formats: .hdr, .exr, .hdri";
        }

        // Emit signal to notify Model of HDR image change
        hdrImageChangeEmitter.fire (hdrFilePath);

        std::ostringstream response;
        response << "HDR image set to: " << hdrPath;
        return response.str();
    }
    catch (const std::exception& e)
    {
        return "Error setting HDR image: " + std::string (e.what());
    }
}

std::string CommandProcessor::processSetContentDirectoryCommand (const std::string& cmd)
{
    // Parse command: "SetContentDirectory path"
    std::istringstream iss (cmd);
    std::string command, contentPath;

    if (!(iss >> command))
    {
        return "Error: Invalid SetContentDirectory format. Expected: SetContentDirectory path";
    }

    // Get the rest of the line as the path (may contain spaces)
    std::getline (iss, contentPath);

    // Trim leading whitespace
    contentPath.erase (0, contentPath.find_first_not_of (" \t"));

    if (contentPath.empty())
    {
        return "Error: Content directory path cannot be empty";
    }

    try
    {
        std::filesystem::path dirPath (contentPath);

        // Check if directory exists
        if (!std::filesystem::exists (dirPath))
        {
            return "Error: Content directory does not exist: " + contentPath;
        }

        if (!std::filesystem::is_directory (dirPath))
        {
            return "Error: Path is not a directory: " + contentPath;
        }

        // Set the content directory property
        properties.renderProps->setValue (RenderKey::ContentFolder, contentPath);

        std::ostringstream response;
        response << "Content directory set to: " << contentPath;
        return response.str();
    }
    catch (const std::exception& e)
    {
        return "Error setting content directory: " + std::string (e.what());
    }
}
std::string CommandProcessor::processPingCommand()
{
    return "pong";
}

std::string CommandProcessor::processCameraInfoCommand()
{
    CameraHandle camera = getCamera();
    if (!camera)
    {
        return "Error: Camera not available";
    }

    // Use your camera's comprehensive data
    Eigen::Vector3f eye = camera->getEyePoint();
    Eigen::Vector3f target = camera->getTarget();
    Eigen::Vector3f up = camera->getWorldUp();
    Eigen::Vector3f viewDir = camera->getViewDirection();
    Eigen::Vector3f right = camera->getRight();
    Eigen::Quaternionf orientation = camera->getOrientation();

    Eigen::Matrix4f viewMatrix = camera->getViewMatrix();

    std::ostringstream response;
    response << "{"
             << "\"eye\":[" << eye.x() << "," << eye.y() << "," << eye.z() << "],"
             << "\"target\":[" << target.x() << "," << target.y() << "," << target.z() << "],"
             << "\"worldUp\":[" << up.x() << "," << up.y() << "," << up.z() << "],"
             << "\"viewDirection\":[" << viewDir.x() << "," << viewDir.y() << "," << viewDir.z() << "],"
             << "\"rightVector\":[" << right.x() << "," << right.y() << "," << right.z() << "],"
             << "\"orientation\":[" << orientation.w() << "," << orientation.x() << "," << orientation.y() << "," << orientation.z() << "],"
             << "\"focalLength\":" << camera->getFocalLength() << ","
             << "\"aperture\":" << camera->getApeture() << ","
             << "\"verticalFOV\":" << (camera->getVerticalFOVradians() * 180.0f / M_PI) << ","
             << "\"viewMatrix\":["
             << viewMatrix (0, 0) << "," << viewMatrix (0, 1) << "," << viewMatrix (0, 2) << "," << viewMatrix (0, 3) << ","
             << viewMatrix (1, 0) << "," << viewMatrix (1, 1) << "," << viewMatrix (1, 2) << "," << viewMatrix (1, 3) << ","
             << viewMatrix (2, 0) << "," << viewMatrix (2, 1) << "," << viewMatrix (2, 2) << "," << viewMatrix (2, 3) << ","
             << viewMatrix (3, 0) << "," << viewMatrix (3, 1) << "," << viewMatrix (3, 2) << "," << viewMatrix (3, 3)
             << "]}";

    return response.str();
}

std::string CommandProcessor::processCameraLookAtCommand (const std::string& cmd)
{
    // Parse command: "SetCameraLookAt eyeX eyeY eyeZ targetX targetY targetZ"
    std::istringstream iss (cmd);
    std::string command;
    float eyeX, eyeY, eyeZ, targetX, targetY, targetZ;

    if (!(iss >> command >> eyeX >> eyeY >> eyeZ >> targetX >> targetY >> targetZ))
    {
        return "Error: Invalid SetCameraLookAt format. Expected: SetCameraLookAt eyeX eyeY eyeZ targetX targetY targetZ";
    }

    CameraHandle camera = getCamera();
    if (!camera)
    {
        return "Error: Camera not available";
    }

    Eigen::Vector3f eye (eyeX, eyeY, eyeZ);
    Eigen::Vector3f target (targetX, targetY, targetZ);

    camera->lookAt (eye, target);

    std::ostringstream response;
    response << "Camera positioned at (" << eyeX << ", " << eyeY << ", " << eyeZ
             << ") looking at (" << targetX << ", " << targetY << ", " << targetZ << ")";
    return response.str();
}

std::string CommandProcessor::processCameraZoomCommand (const std::string& cmd)
{
    // Parse command: "CameraZoom distance"
    std::istringstream iss (cmd);
    std::string command;
    float distance;

    if (!(iss >> command >> distance))
    {
        return "Error: Invalid CameraZoom format. Expected: CameraZoom distance";
    }

    CameraHandle camera = getCamera();
    if (!camera)
    {
        return "Error: Camera not available";
    }

    camera->zoom (distance);

    std::ostringstream response;
    response << "Camera zoomed by " << distance << " units";
    return response.str();
}

std::string CommandProcessor::processCameraPanCommand (const std::string& cmd)
{
    // Parse command: "CameraPan horizontal vertical"
    std::istringstream iss (cmd);
    std::string command;
    float horizontal, vertical;

    if (!(iss >> command >> horizontal >> vertical))
    {
        return "Error: Invalid CameraPan format. Expected: CameraPan horizontal vertical";
    }

    CameraHandle camera = getCamera();
    if (!camera)
    {
        return "Error: Camera not available";
    }

    camera->panHorizontal (horizontal);
    camera->panVertical (vertical);

    std::ostringstream response;
    response << "Camera panned horizontally by " << horizontal
             << " and vertically by " << vertical;
    return response.str();
}

std::string CommandProcessor::processBackgroundColorCommand (const std::string& cmd)
{
    // Parse command: "SetBackgroundColor r g b a"
    std::istringstream iss (cmd);
    std::string command;
    int r, g, b, a = 255;

    if (!(iss >> command >> r >> g >> b))
    {
        return "Error: Invalid SetBackgroundColor format. Expected: SetBackgroundColor r g b [a]";
    }

    // Alpha is optional, default to 255 if not provided
    iss >> a;

    // Validate color values
    if (!validateColorValues (r, g, b, a))
    {
        return "Error: Color values must be between 0 and 255";
    }

    // Apply the background color change
    if (gui && gui->getCanvas())
    {
        Vector4i color (r, g, b, a);
        gui->getCanvas()->set_background_color (color);

        std::ostringstream response;
        response << "Background color set to RGB(" << r << ", " << g << ", " << b << ", " << a << ")";
        return response.str();
    }
    else
    {
        return "Error: Canvas not available";
    }
}

std::string CommandProcessor::processCubeRotationCommand (const std::string& cmd)
{
    // Parse command: "SetCubeRotation angle_degrees auto_rotate"
    std::istringstream iss (cmd);
    std::string command;
    float angleDegrees;
    int autoRotate;

    if (!(iss >> command >> angleDegrees >> autoRotate))
    {
        return "Error: Invalid SetCubeRotation format. Expected: SetCubeRotation angle_degrees auto_rotate";
    }

    // Convert degrees to radians
    float angleRadians = angleDegrees * (3.14159f / 180.0f);

    // Apply the rotation change
    if (gui && gui->getCanvas())
    {
        // Note: These methods appear to be commented out in the original
        // gui->getCanvas()->set_rotation(angleRadians);
        // gui->getCanvas()->set_auto_rotate(autoRotate != 0);

        std::ostringstream response;
        response << "Cube rotation set to " << angleDegrees << " degrees";
        if (autoRotate)
            response << " with auto-rotation enabled";
        else
            response << " with auto-rotation disabled";

        return response.str();
    }
    else
    {
        return "Error: Canvas not available";
    }
}

bool CommandProcessor::validateColorValues (int r, int g, int b, int a)
{
    return (r >= 0 && r <= 255 &&
            g >= 0 && g <= 255 &&
            b >= 0 && b <= 255 &&
            a >= 0 && a <= 255);
}

CameraHandle CommandProcessor::getCamera()
{
    if (gui && gui->getCamera())
    {
        return gui->getCamera();
    }
    return nullptr;
}

std::string CommandProcessor::processSetPipelineCommand (const std::string& cmd)
{
    std::istringstream iss (cmd);
    std::string command;
    std::string pipelineName;

    if (!(iss >> command >> pipelineName))
    {
        return "Error: Invalid SetPipeline format. Expected: SetPipeline pipeline_name";
    }

    // Fire the pipeline change event through the UI
    if (gui)
    {
        gui->onPipelineChange.fire(pipelineName);
        return "Pipeline set to: " + pipelineName;
    }
    else
    {
        return "Error: GUI not available";
    }
}

std::string CommandProcessor::processGetAvailablePipelinesCommand()
{
    // For now, return the hardcoded list. In a full implementation,
    // this could query the backend for available pipelines
    return "Available pipelines: realtime, quality, sequential";
}