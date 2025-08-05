#pragma once

#include "../oiio_core/oiio_core.h"
#include "../wabi_core/wabi_core.h"

#include <fastgltf/simdjson.h>
#include <fastgltf/core.hpp>
#include <fastgltf/tools.hpp>

#include <cereal/cereal.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/string.hpp>

constexpr float DEFAULT_ZOOM_FACTOR = 0.5f;
constexpr float DEFAULT_ZOOM_MULTIPLIER = 200.0f;

const Eigen::Vector3f DEFAULT_CAMERA_POSIIION = Eigen::Vector3f (2.0f, 3.0f, -4.0f);
const Eigen::Vector3f DEFAULT_CAMERA_TARGET = Eigen::Vector3f::Zero();
constexpr float DEFAULT_FOV_DEGREES = 45.0f;
const float DEFAULT_ASPECT = (float)DEFAULT_DESKTOP_WINDOW_WIDTH / (float)DEFAULT_DESKTOP_WINDOW_HEIGHT;
constexpr float DEFAULT_NEAR_PLANE = 0.01f;
constexpr float DEFAULT_FAR_PLANE = 1000.0f;
constexpr float DEFAULT_FOCAL_LENGTH = 1.0f;
constexpr float DEFAULT_APETURE = 0.0f;

namespace cereal
{
    template <class Archive>
    void serialize (Archive& ar, Eigen::Vector3f& vector)
    {
        ar (cereal::make_nvp ("x", vector.x()),
            cereal::make_nvp ("y", vector.y()),
            cereal::make_nvp ("z", vector.z()));
    }
} // namespace cereal

// rendering transforms and datat
struct RenderT
{
    Eigen::Matrix4f model;
    Eigen::Matrix4f proj;
    Eigen::Matrix4f view;
    Eigen::Vector2i screenSize;
};

struct KeyFrame
{
    float time = 0.0f;
    Eigen::Vector3f translation = Eigen::Vector3f::Zero();
    Eigen::Vector3f scale = Eigen::Vector3f::Ones();
    Eigen::Quaternionf rotation = Eigen::Quaternionf::Identity();

    // Debug function to log KeyFrame information
    void debug() const
    {
        std::ostringstream ss;
        ss << "KeyFrame: "
           << "Time=" << time
           << ", Translation=(" << translation.x() << ", " << translation.y() << ", " << translation.z() << ")"
           << ", Scale=(" << scale.x() << ", " << scale.y() << ", " << scale.z() << ")"
           << ", Rotation=(" << rotation.x() << ", " << rotation.y() << ", " << rotation.z() << ", " << rotation.w() << ")";

        LOG (DBUG) << ss.str();
    }
};

struct AnimationChannel
{
    std::string targetNode;
    fastgltf::AnimationPath path;
    std::vector<KeyFrame> keyFrames;
};

struct Animation
{
    std::string name;
    std::vector<AnimationChannel> channels;
};

namespace sabi
{
#include "excludeFromBuild/scene/RenderableData.h"

    // cgModel
#include "excludeFromBuild/cgmodel/MeshOptions.h"
// #include "excludeFromBuild/model/GltfMaterial.h"
// #include "excludeFromBuild/model/cgModelSurface.h"
// #include "excludeFromBuild/model/cgModel.h"

// camera
#include "excludeFromBuild/camera/CameraSensor.h"
#include "excludeFromBuild/camera/CameraBody.h"

    // cppGltf
#include "excludeFromBuild/io/GLTFUtil.h"
#include "excludeFromBuild/cgmodel/CgMaterial.h"
#include "excludeFromBuild/cgmodel/CgModelSurface.h"
#include "excludeFromBuild/cgmodel/CgModel.h"

    // scene
#include "excludeFromBuild/scene/RayIntersectionInfo.h"
#include "excludeFromBuild/scene/SpaceTime.h"
#include "excludeFromBuild/scene/RenderableState.h"
#include "excludeFromBuild/scene/RenderableDesc.h"
#include "excludeFromBuild/scene/Renderable.h"
#include "excludeFromBuild/scene/WorldItem.h"
#include "excludeFromBuild/scene/WorldComposite.h"
#include "excludeFromBuild/scene/SceneOptions.h"

// tools
#include "excludeFromBuild/tools/LoadStrategy.h"
#include "excludeFromBuild/tools/MeshOps.h"
#include "excludeFromBuild/tools/NormalizedClump.h"
#include "excludeFromBuild/tools/RadialFlower.h"

#include "excludeFromBuild/lwo3/LWO3Defs.h"
#include "excludeFromBuild/lwo3/LWO3Element.h"
#include "excludeFromBuild/lwo3/LWO3Visitor.h"
#include "excludeFromBuild/lwo3/LWO3Chunk.h"
#include "excludeFromBuild/lwo3/LWO3Form.h"

#include "excludeFromBuild/io/LWO3Navigator.h"
#include "excludeFromBuild/io/LWO3Tree.h"
#include "excludeFromBuild/io/LWO3NodeData.h"
#include "excludeFromBuild/io/LWO3NodeGraph.h"
#include "excludeFromBuild/io/LWO3Surface.h"
#include "excludeFromBuild/io/LWO3Layer.h"
#include "excludeFromBuild/io/LWO3Reader.h"
#include "excludeFromBuild/io/LWO3ToCgModelConverter.h"
#include "excludeFromBuild/io/LWO3Material.h"
#include "excludeFromBuild/io/LWO3MaterialManager.h"
} // namespace sabi

// must be outside sabi
#include "excludeFromBuild/io/GltfAnimationExporter.h"
#include "excludeFromBuild/io/GLTFImporter.h"
#include "excludeFromBuild/animation/AnimationBuilder.h"

#include "excludeFromBuild/io/AssetPathManager.h"

// #include "excludeFromBuild/io/LWO3ToCgModel.h"