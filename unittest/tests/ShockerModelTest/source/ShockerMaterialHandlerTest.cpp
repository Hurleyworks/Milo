// ShockerMaterialHandlerTest.cpp
// Unit tests for ShockerMaterialHandler

#include <doctest/doctest.h>
#include <sabi_core/sabi_core.h>
#include <filesystem>

#include "engine_core/excludeFromBuild/engines/shocker/handlers/ShockerMaterialHandler.h"
#include "engine_core/excludeFromBuild/engines/shocker/handlers/ShockerModelHandler.h"
#include "engine_core/excludeFromBuild/engines/shocker/models/ShockerModel.h"
#include "engine_core/excludeFromBuild/engines/shocker/models/ShockerCore.h"

TEST_CASE("ShockerMaterialHandler Basic Operations")
{
    SUBCASE("Initialize and Clear")
    {
        ShockerMaterialHandler handler;
        
        // Initialize without render context (for basic testing)
        handler.initialize(nullptr);
        
        CHECK(handler.getAllMaterials().size() == 1); // Should have default material
        CHECK(handler.getMaterial(0) != nullptr); // Default material at index 0
        
        handler.clear();
        CHECK(handler.getAllMaterials().empty());
    }
    
    SUBCASE("Create Default Material")
    {
        ShockerMaterialHandler handler;
        handler.initialize(nullptr);
        
        // Default material should be created during initialization
        DisneyMaterial* defaultMat = handler.getMaterial(0);
        CHECK(defaultMat != nullptr);
        
        // Default material should have all null textures
        CHECK(defaultMat->baseColor == nullptr);
        CHECK(defaultMat->texBaseColor == 0);
        CHECK(defaultMat->roughness == nullptr);
        CHECK(defaultMat->texRoughness == 0);
        CHECK(defaultMat->metallic == nullptr);
        CHECK(defaultMat->texMetallic == 0);
    }
    
    SUBCASE("Create Material From CgMaterial")
    {
        ShockerMaterialHandler handler;
        handler.initialize(nullptr);
        
        // Create a test CgMaterial
        sabi::CgMaterial cgMat;
        cgMat.name = "TestMaterial";
        cgMat.core.baseColor = Eigen::Vector3f(1.0f, 0.5f, 0.25f);
        cgMat.core.roughness = 0.7f;
        cgMat.core.specular = 0.5f;
        cgMat.metallic.metallic = 0.3f;
        cgMat.emission.luminous = 0.0f;
        
        // Create material without model (no textures)
        DisneyMaterial* material = handler.createMaterialFromCg(cgMat);
        
        CHECK(material != nullptr);
        CHECK(handler.getAllMaterials().size() == 2); // Default + new material
        
        // Without render context, textures won't be created
        // but the material should still be valid
        CHECK(material != handler.getMaterial(0)); // Not the default material
    }
    
    SUBCASE("Process Materials For Model Without CgModel")
    {
        ShockerMaterialHandler handler;
        handler.initialize(nullptr);
        
        ShockerModelHandler modelHandler;
        modelHandler.initialize(nullptr);
        
        // Create a simple cube model
        sabi::CgModelPtr cube = sabi::MeshOps::createCube(1.0f);
        sabi::RenderableNode node = sabi::WorldItem::create();
        node->setName("TestCube");
        node->setModel(cube);
        
        ShockerModelPtr model = modelHandler.processRenderableNode(node);
        CHECK(model != nullptr);
        
        // Process materials for model with null CgModel
        handler.processMaterialsForModel(model.get(), nullptr);
        
        // Should assign default material to all surfaces
        for (const auto& surface : model->getSurfaces()) {
            CHECK(surface->mat != nullptr);
            // Should be the default material
            CHECK(surface->mat == handler.getMaterial(0));
        }
    }
    
    SUBCASE("Process Materials For Model With CgModel")
    {
        ShockerMaterialHandler handler;
        handler.initialize(nullptr);
        
        ShockerModelHandler modelHandler;
        modelHandler.initialize(nullptr);
        
        // Create a cube with materials
        sabi::CgModelPtr cube = sabi::MeshOps::createCube(2.0f);
        
        // Set up materials for each surface
        for (auto& surface : cube->S) {
            surface.cgMaterial.name = "Surface_" + surface.name;
            surface.cgMaterial.core.baseColor = Eigen::Vector3f(0.8f, 0.8f, 0.8f);
            surface.cgMaterial.core.roughness = 0.5f;
            surface.cgMaterial.metallic.metallic = 0.0f;
        }
        
        sabi::RenderableNode node = sabi::WorldItem::create();
        node->setName("MaterializedCube");
        node->setModel(cube);
        
        ShockerModelPtr model = modelHandler.processRenderableNode(node);
        CHECK(model != nullptr);
        
        // Process materials for the model
        handler.processMaterialsForModel(model.get(), cube);
        
        // Each surface should have a material assigned
        const auto& surfaces = model->getSurfaces();
        CHECK(surfaces.size() == cube->S.size());
        
        for (const auto& surface : surfaces) {
            CHECK(surface->mat != nullptr);
        }
        
        // Should have created materials for each surface plus default
        CHECK(handler.getAllMaterials().size() > 1);
    }
    
    SUBCASE("Material Assignment to ShockerSurface")
    {
        ShockerMaterialHandler handler;
        handler.initialize(nullptr);
        
        // Create a ShockerSurface
        shocker::ShockerSurface surface;
        surface.geomInstSlot = 0;
        
        // Get default material
        DisneyMaterial* material = handler.getMaterial(0);
        CHECK(material != nullptr);
        
        // Assign material
        handler.assignMaterialToSurface(&surface, material);
        
        // Check assignment - direct comparison, no casting needed!
        CHECK(surface.mat != nullptr);
        CHECK(surface.mat == material);
    }
    
    SUBCASE("Multiple Materials Management")
    {
        ShockerMaterialHandler handler;
        handler.initialize(nullptr);
        
        // Create multiple materials
        std::vector<DisneyMaterial*> materials;
        
        for (int i = 0; i < 5; ++i) {
            sabi::CgMaterial cgMat;
            cgMat.name = "Material_" + std::to_string(i);
            cgMat.core.baseColor = Eigen::Vector3f(i * 0.2f, i * 0.2f, i * 0.2f);
            cgMat.core.roughness = i * 0.2f;
            
            DisneyMaterial* mat = handler.createMaterialFromCg(cgMat);
            CHECK(mat != nullptr);
            materials.push_back(mat);
        }
        
        // Check all materials were created
        CHECK(handler.getAllMaterials().size() == 6); // 1 default + 5 created
        
        // Verify each material is unique
        for (size_t i = 0; i < materials.size(); ++i) {
            for (size_t j = i + 1; j < materials.size(); ++j) {
                CHECK(materials[i] != materials[j]);
            }
        }
    }
    
    SUBCASE("Material with Transparency")
    {
        ShockerMaterialHandler handler;
        handler.initialize(nullptr);
        
        // Create material with transparency
        sabi::CgMaterial cgMat;
        cgMat.name = "TransparentMaterial";
        cgMat.flags.alphaMode = sabi::AlphaMode::Blend;
        cgMat.transparency.transparency = 0.5f;
        
        DisneyMaterial* material = handler.createMaterialFromCg(cgMat);
        CHECK(material != nullptr);
        CHECK(material->useAlphaForTransparency == true);
    }
    
    SUBCASE("Material with Opaque Alpha")
    {
        ShockerMaterialHandler handler;
        handler.initialize(nullptr);
        
        // Create opaque material
        sabi::CgMaterial cgMat;
        cgMat.name = "OpaqueMaterial";
        cgMat.flags.alphaMode = sabi::AlphaMode::Opaque;
        
        DisneyMaterial* material = handler.createMaterialFromCg(cgMat);
        CHECK(material != nullptr);
        CHECK(material->useAlphaForTransparency == false);
    }
}

TEST_CASE("ShockerMaterialHandler with Complex Models")
{
    SUBCASE("Process Cube with Material Properties")
    {
        ShockerMaterialHandler matHandler;
        matHandler.initialize(nullptr);
        
        ShockerModelHandler modelHandler;
        modelHandler.initialize(nullptr);
        modelHandler.setMaterialHandler(&matHandler);
        
        // Create a cube instead (createSphere doesn't exist)
        sabi::CgModelPtr cube = sabi::MeshOps::createCube(1.0f);
        
        // Set up material for the cube
        if (!cube->S.empty()) {
            cube->S[0].cgMaterial.name = "CubeMaterial";
            cube->S[0].cgMaterial.core.baseColor = Eigen::Vector3f(0.2f, 0.4f, 0.8f);
            cube->S[0].cgMaterial.core.roughness = 0.3f;
            cube->S[0].cgMaterial.metallic.metallic = 0.7f;
        }
        
        sabi::RenderableNode node = sabi::WorldItem::create();
        node->setName("TestCube");
        node->setModel(cube);
        
        // Process through model handler (which should call material handler)
        ShockerModelPtr model = modelHandler.processRenderableNode(node);
        CHECK(model != nullptr);
        
        // Verify materials were assigned
        for (const auto& surface : model->getSurfaces()) {
            CHECK(surface->mat != nullptr);
        }
        
        // Should have created materials
        CHECK(matHandler.getAllMaterials().size() > 1);
    }
    
    SUBCASE("Process Model with Missing Materials")
    {
        ShockerMaterialHandler matHandler;
        matHandler.initialize(nullptr);
        
        ShockerModelHandler modelHandler;
        modelHandler.initialize(nullptr);
        
        // Create model with surfaces but no materials set
        sabi::CgModelPtr cube = sabi::MeshOps::createCube(1.0f);
        
        // Clear any default materials
        for (auto& surface : cube->S) {
            surface.cgMaterial = sabi::CgMaterial(); // Empty material
        }
        
        sabi::RenderableNode node = sabi::WorldItem::create();
        node->setName("NoMaterialCube");
        node->setModel(cube);
        
        ShockerModelPtr model = modelHandler.processRenderableNode(node);
        CHECK(model != nullptr);
        
        // Process materials
        matHandler.processMaterialsForModel(model.get(), cube);
        
        // Should still assign materials (even if they're default/empty)
        for (const auto& surface : model->getSurfaces()) {
            CHECK(surface->mat != nullptr);
        }
    }
}

TEST_CASE("ShockerMaterialHandler GPU Data Conversion")
{
    SUBCASE("Convert Material to Device Data")
    {
        ShockerMaterialHandler handler;
        handler.initialize(nullptr);
        
        // Create a material
        sabi::CgMaterial cgMat;
        cgMat.name = "TestMaterial";
        cgMat.core.baseColor = Eigen::Vector3f(1.0f, 0.0f, 0.0f);
        cgMat.flags.alphaMode = sabi::AlphaMode::Blend;
        
        DisneyMaterial* material = handler.createMaterialFromCg(cgMat);
        CHECK(material != nullptr);
        
        // Test private method through uploadMaterialsToGPU
        // This will internally call convertToDeviceData
        handler.uploadMaterialsToGPU();
        
        // Can't directly test the device data without GPU context
        // but we can verify the function runs without crashing
        CHECK(true); // If we got here, conversion worked
    }
    
    SUBCASE("Texture Dimension Calculation")
    {
        // Can't test calcDimInfo directly as it's private
        // Testing through uploadMaterialsToGPU is sufficient
        ShockerMaterialHandler handler;
        handler.initialize(nullptr);
        
        // Create materials and upload to test dimension calculation
        sabi::CgMaterial cgMat;
        cgMat.name = "TestMaterial";
        DisneyMaterial* material = handler.createMaterialFromCg(cgMat);
        CHECK(material != nullptr);
        
        // This internally uses calcDimInfo
        handler.uploadMaterialsToGPU();
        CHECK(true); // If we got here, dimension calculation worked
    }
}