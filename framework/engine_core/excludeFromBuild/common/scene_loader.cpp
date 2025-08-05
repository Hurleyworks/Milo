#include "scene_loader.h"
#include <fstream>
#include <iostream>
#include <stdexcept>

JsonScene SceneLoader::LoadFromJSON(const std::filesystem::path& json_path) {
    if (!std::filesystem::exists(json_path)) {
        throw std::runtime_error("JsonScene file not found: " + json_path.string());
    }
    
    std::ifstream file(json_path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open scene file: " + json_path.string());
    }
    
    json j;
    file >> j;
    
    JsonScene scene;
    
    // Parse camera
    if (j.contains("camera")) {
        scene.camera = ParseCamera(j["camera"]);
    }
    
    // Parse environment
    if (j.contains("environment")) {
        scene.environment = ParseEnvironment(j["environment"]);
    }
    
    // Parse render settings
    if (j.contains("render_settings")) {
        scene.render_settings = ParseRenderSettings(j["render_settings"]);
    }
    
    // Parse meshes
    if (j.contains("meshes")) {
        for (const auto& mesh_json : j["meshes"]) {
            scene.meshes.push_back(ParseMesh(mesh_json));
        }
    }
    
    // Parse primitives
    if (j.contains("primitives")) {
        for (const auto& prim_json : j["primitives"]) {
            scene.primitives.push_back(ParsePrimitive(prim_json));
        }
    }
    
    // Parse instances
    if (j.contains("instances")) {
        for (const auto& inst_json : j["instances"]) {
            scene.instances.push_back(ParseInstance(inst_json));
        }
    }
    
    return scene;
}

void SceneLoader::SaveToJSON(const JsonScene& scene, const std::filesystem::path& json_path) {
    json j;
    
    j["camera"] = SerializeCamera(scene.camera);
    j["environment"] = SerializeEnvironment(scene.environment);
    j["render_settings"] = SerializeRenderSettings(scene.render_settings);
    
    j["meshes"] = json::array();
    for (const auto& mesh : scene.meshes) {
        j["meshes"].push_back(SerializeMesh(mesh));
    }
    
    j["primitives"] = json::array();
    for (const auto& prim : scene.primitives) {
        j["primitives"].push_back(SerializePrimitive(prim));
    }
    
    j["instances"] = json::array();
    for (const auto& inst : scene.instances) {
        j["instances"].push_back(SerializeInstance(inst));
    }
    
    std::ofstream file(json_path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not create scene file: " + json_path.string());
    }
    
    file << j.dump(2);
}

Point3D SceneLoader::ParsePoint3D(const json& j) {
    if (j.is_array() && j.size() >= 3) {
        return {j[0].get<float>(), j[1].get<float>(), j[2].get<float>()};
    }
    return {0.0f, 0.0f, 0.0f};
}

RGB SceneLoader::ParseRGB(const json& j) {
    if (j.is_array() && j.size() >= 3) {
        return {j[0].get<float>(), j[1].get<float>(), j[2].get<float>()};
    }
    return {1.0f, 1.0f, 1.0f};
}

SceneCamera SceneLoader::ParseCamera(const json& j) {
    SceneCamera camera;
    
    if (j.contains("position")) camera.position = ParsePoint3D(j["position"]);
    if (j.contains("roll")) camera.roll = j["roll"].get<float>();
    if (j.contains("pitch")) camera.pitch = j["pitch"].get<float>();
    if (j.contains("yaw")) camera.yaw = j["yaw"].get<float>();
    if (j.contains("brightness")) camera.brightness = j["brightness"].get<float>();
    
    return camera;
}

SceneEnvironment SceneLoader::ParseEnvironment(const json& j) {
    SceneEnvironment env;
    
    if (j.contains("texture_path")) env.texture_path = j["texture_path"].get<std::string>();
    if (j.contains("power_coefficient")) env.power_coefficient = j["power_coefficient"].get<float>();
    if (j.contains("rotation")) env.rotation = j["rotation"].get<float>();
    if (j.contains("enabled")) env.enabled = j["enabled"].get<bool>();
    
    return env;
}

SceneMaterial SceneLoader::ParseMaterial(const json& j) {
    SceneMaterial mat;
    
    if (j.contains("type")) mat.type = j["type"].get<std::string>();
    if (j.contains("albedo")) mat.albedo = ParseRGB(j["albedo"]);
    if (j.contains("emittance")) mat.emittance = ParseRGB(j["emittance"]);
    if (j.contains("roughness")) mat.roughness = j["roughness"].get<float>();
    if (j.contains("metallic")) mat.metallic = j["metallic"].get<float>();
    if (j.contains("albedo_texture")) mat.albedo_texture = j["albedo_texture"].get<std::string>();
    if (j.contains("normal_texture")) mat.normal_texture = j["normal_texture"].get<std::string>();
    
    return mat;
}

SceneAnimation SceneLoader::ParseAnimation(const json& j) {
    SceneAnimation anim;
    
    if (j.contains("begin_position")) anim.begin_position = ParsePoint3D(j["begin_position"]);
    if (j.contains("end_position")) anim.end_position = ParsePoint3D(j["end_position"]);
    if (j.contains("begin_roll")) anim.begin_roll = j["begin_roll"].get<float>();
    if (j.contains("begin_pitch")) anim.begin_pitch = j["begin_pitch"].get<float>();
    if (j.contains("begin_yaw")) anim.begin_yaw = j["begin_yaw"].get<float>();
    if (j.contains("end_roll")) anim.end_roll = j["end_roll"].get<float>();
    if (j.contains("end_pitch")) anim.end_pitch = j["end_pitch"].get<float>();
    if (j.contains("end_yaw")) anim.end_yaw = j["end_yaw"].get<float>();
    if (j.contains("begin_scale")) anim.begin_scale = j["begin_scale"].get<float>();
    if (j.contains("end_scale")) anim.end_scale = j["end_scale"].get<float>();
    if (j.contains("frequency")) anim.frequency = j["frequency"].get<float>();
    if (j.contains("initial_time")) anim.initial_time = j["initial_time"].get<float>();
    
    return anim;
}

SceneMesh SceneLoader::ParseMesh(const json& j) {
    SceneMesh mesh;
    
    if (j.contains("name")) mesh.name = j["name"].get<std::string>();
    if (j.contains("file_path")) mesh.file_path = j["file_path"].get<std::string>();
    if (j.contains("scale")) mesh.scale = j["scale"].get<float>();
    if (j.contains("material_convention")) mesh.material_convention = j["material_convention"].get<std::string>();
    if (j.contains("material")) mesh.material = ParseMaterial(j["material"]);
    
    return mesh;
}

ScenePrimitive SceneLoader::ParsePrimitive(const json& j) {
    ScenePrimitive prim;
    
    if (j.contains("type")) prim.type = j["type"].get<std::string>();
    if (j.contains("name")) prim.name = j["name"].get<std::string>();
    if (j.contains("dim_x")) prim.dim_x = j["dim_x"].get<float>();
    if (j.contains("dim_z")) prim.dim_z = j["dim_z"].get<float>();
    if (j.contains("material")) prim.material = ParseMaterial(j["material"]);
    
    return prim;
}

SceneInstance SceneLoader::ParseInstance(const json& j) {
    SceneInstance inst;
    
    if (j.contains("mesh_name")) inst.mesh_name = j["mesh_name"].get<std::string>();
    if (j.contains("name")) inst.name = j["name"].get<std::string>();
    if (j.contains("position")) inst.position = ParsePoint3D(j["position"]);
    if (j.contains("roll")) inst.roll = j["roll"].get<float>();
    if (j.contains("pitch")) inst.pitch = j["pitch"].get<float>();
    if (j.contains("yaw")) inst.yaw = j["yaw"].get<float>();
    if (j.contains("scale")) inst.scale = j["scale"].get<float>();
    if (j.contains("animation")) {
        inst.animation = ParseAnimation(j["animation"]);
        inst.has_animation = true;
    }
    
    return inst;
}

SceneRenderSettings SceneLoader::ParseRenderSettings(const json& j) {
    SceneRenderSettings settings;
    
    if (j.contains("max_path_length")) settings.max_path_length = j["max_path_length"].get<int>();
    if (j.contains("enable_jittering")) settings.enable_jittering = j["enable_jittering"].get<bool>();
    if (j.contains("enable_bump_mapping")) settings.enable_bump_mapping = j["enable_bump_mapping"].get<bool>();
    if (j.contains("enable_debug_print")) settings.enable_debug_print = j["enable_debug_print"].get<bool>();
    if (j.contains("width")) settings.width = j["width"].get<int>();
    if (j.contains("height")) settings.height = j["height"].get<int>();
    
    return settings;
}

json SceneLoader::SerializePoint3D(const Point3D& p) {
    return json::array({p.x, p.y, p.z});
}

json SceneLoader::SerializeRGB(const RGB& rgb) {
    return json::array({rgb.r, rgb.g, rgb.b});
}

json SceneLoader::SerializeCamera(const SceneCamera& camera) {
    json j;
    j["position"] = SerializePoint3D(camera.position);
    j["roll"] = camera.roll;
    j["pitch"] = camera.pitch;
    j["yaw"] = camera.yaw;
    j["brightness"] = camera.brightness;
    return j;
}

json SceneLoader::SerializeEnvironment(const SceneEnvironment& env) {
    json j;
    j["texture_path"] = env.texture_path.string();
    j["power_coefficient"] = env.power_coefficient;
    j["rotation"] = env.rotation;
    j["enabled"] = env.enabled;
    return j;
}

json SceneLoader::SerializeMaterial(const SceneMaterial& mat) {
    json j;
    j["type"] = mat.type;
    j["albedo"] = SerializeRGB(mat.albedo);
    j["emittance"] = SerializeRGB(mat.emittance);
    j["roughness"] = mat.roughness;
    j["metallic"] = mat.metallic;
    if (!mat.albedo_texture.empty()) j["albedo_texture"] = mat.albedo_texture.string();
    if (!mat.normal_texture.empty()) j["normal_texture"] = mat.normal_texture.string();
    return j;
}

json SceneLoader::SerializeAnimation(const SceneAnimation& anim) {
    json j;
    j["begin_position"] = SerializePoint3D(anim.begin_position);
    j["end_position"] = SerializePoint3D(anim.end_position);
    j["begin_roll"] = anim.begin_roll;
    j["begin_pitch"] = anim.begin_pitch;
    j["begin_yaw"] = anim.begin_yaw;
    j["end_roll"] = anim.end_roll;
    j["end_pitch"] = anim.end_pitch;
    j["end_yaw"] = anim.end_yaw;
    j["begin_scale"] = anim.begin_scale;
    j["end_scale"] = anim.end_scale;
    j["frequency"] = anim.frequency;
    j["initial_time"] = anim.initial_time;
    return j;
}

json SceneLoader::SerializeMesh(const SceneMesh& mesh) {
    json j;
    j["name"] = mesh.name;
    j["file_path"] = mesh.file_path.string();
    j["scale"] = mesh.scale;
    j["material_convention"] = mesh.material_convention;
    j["material"] = SerializeMaterial(mesh.material);
    return j;
}

json SceneLoader::SerializePrimitive(const ScenePrimitive& prim) {
    json j;
    j["type"] = prim.type;
    j["name"] = prim.name;
    j["dim_x"] = prim.dim_x;
    j["dim_z"] = prim.dim_z;
    j["material"] = SerializeMaterial(prim.material);
    return j;
}

json SceneLoader::SerializeInstance(const SceneInstance& inst) {
    json j;
    j["mesh_name"] = inst.mesh_name;
    j["name"] = inst.name;
    j["position"] = SerializePoint3D(inst.position);
    j["roll"] = inst.roll;
    j["pitch"] = inst.pitch;
    j["yaw"] = inst.yaw;
    j["scale"] = inst.scale;
    if (inst.has_animation) {
        j["animation"] = SerializeAnimation(inst.animation);
    }
    return j;
}

json SceneLoader::SerializeRenderSettings(const SceneRenderSettings& settings) {
    json j;
    j["max_path_length"] = settings.max_path_length;
    j["enable_jittering"] = settings.enable_jittering;
    j["enable_bump_mapping"] = settings.enable_bump_mapping;
    j["enable_debug_print"] = settings.enable_debug_print;
    j["width"] = settings.width;
    j["height"] = settings.height;
    return j;
}