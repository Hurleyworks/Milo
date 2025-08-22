#pragma once

#include "basic_types.h"
#include "../json.hpp"
#include <string>
#include <vector>
#include <map>
#include <filesystem>

using json = nlohmann::json;

struct SceneCamera {
    Point3D position = {0.0f, 0.0f, 0.0f};
    float roll = 0.0f;
    float pitch = 0.0f; 
    float yaw = 0.0f;
    float brightness = 0.0f;
};

struct SceneEnvironment {
    std::filesystem::path texture_path;
    float power_coefficient = 0.0f;
    float rotation = 0.0f;
    bool enabled = true;
};

struct SceneMaterial {
    std::string type = "lambert"; // "lambert", "simple_pbr", etc.
    RGB albedo = {0.8f, 0.8f, 0.8f};
    RGB emittance = {0.0f, 0.0f, 0.0f};
    float roughness = 0.5f;
    float metallic = 0.0f;
    std::filesystem::path albedo_texture;
    std::filesystem::path normal_texture;
};

struct SceneAnimation {
    Point3D begin_position = {0.0f, 0.0f, 0.0f};
    Point3D end_position = {0.0f, 0.0f, 0.0f};
    float begin_roll = 0.0f, begin_pitch = 0.0f, begin_yaw = 0.0f;
    float end_roll = 0.0f, end_pitch = 0.0f, end_yaw = 0.0f;
    float begin_scale = 1.0f;
    float end_scale = 1.0f;
    float frequency = 1.0f;
    float initial_time = 0.0f;
};

struct SceneMesh {
    std::string name;
    std::filesystem::path file_path;
    float scale = 1.0f;
    std::string material_convention = "trad"; // "trad" or "simple_pbr"
    SceneMaterial material;
};

struct ScenePrimitive {
    std::string type; // "rectangle"
    std::string name;
    // Rectangle specific
    float dim_x = 1.0f;
    float dim_z = 1.0f;
    SceneMaterial material;
};

struct SceneInstance {
    std::string mesh_name;
    std::string name;
    Point3D position = {0.0f, 0.0f, 0.0f};
    float roll = 0.0f, pitch = 0.0f, yaw = 0.0f;
    float scale = 1.0f;
    SceneAnimation animation;
    bool has_animation = false;
};

struct SceneRenderSettings {
    int max_path_length = 5;
    bool enable_jittering = false;
    bool enable_bump_mapping = false;
    bool enable_debug_print = false;
    int width = 1920;
    int height = 1080;
};

struct JsonScene {
    SceneCamera camera;
    SceneEnvironment environment;
    SceneRenderSettings render_settings;
    std::vector<SceneMesh> meshes;
    std::vector<ScenePrimitive> primitives;
    std::vector<SceneInstance> instances;
};

class SceneLoader {
public:
    static JsonScene LoadFromJSON(const std::filesystem::path& json_path);
    static void SaveToJSON(const JsonScene& scene, const std::filesystem::path& json_path);
    
private:
    static Point3D ParsePoint3D(const json& j);
    static RGB ParseRGB(const json& j);
    static SceneCamera ParseCamera(const json& j);
    static SceneEnvironment ParseEnvironment(const json& j);
    static SceneMaterial ParseMaterial(const json& j);
    static SceneAnimation ParseAnimation(const json& j);
    static SceneMesh ParseMesh(const json& j);
    static ScenePrimitive ParsePrimitive(const json& j);
    static SceneInstance ParseInstance(const json& j);
    static SceneRenderSettings ParseRenderSettings(const json& j);
    
    static json SerializePoint3D(const Point3D& p);
    static json SerializeRGB(const RGB& rgb);
    static json SerializeCamera(const SceneCamera& camera);
    static json SerializeEnvironment(const SceneEnvironment& env);
    static json SerializeMaterial(const SceneMaterial& mat);
    static json SerializeAnimation(const SceneAnimation& anim);
    static json SerializeMesh(const SceneMesh& mesh);
    static json SerializePrimitive(const ScenePrimitive& prim);
    static json SerializeInstance(const SceneInstance& inst);
    static json SerializeRenderSettings(const SceneRenderSettings& settings);
};