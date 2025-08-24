#pragma once

#include "scene/Renderable.h"
#include "common_shared.h"
#include "common_host.h"
#include "cgmodel/cgModel.h"
#include "cgmodel/cgModelSurface.h"
#include <cuda_runtime.h>
#include <optix.h>
#include <vector>
#include <unordered_map>
#include <memory>

// Forward declarations
namespace optixu {
    template<typename T> class Buffer;
}

// Class to convert RenderableNode scene graph to OptiX device structures
class RenderableNodeConverter {
public:
    struct DeviceBuffers {
        // Material data
        std::vector<shared::MaterialData> materials;
        CUdeviceptr materialBuffer = 0;
        
        // Geometry instance data
        std::vector<shared::GeometryInstanceData> geometryInstances;
        CUdeviceptr geometryInstanceBuffer = 0;
        
        // Instance data (transforms)
        std::vector<shared::InstanceData> instances;
        CUdeviceptr instanceBuffer = 0;
        
        // Vertex and triangle buffers per geometry
        struct GeometryBuffers {
            std::vector<shared::Vertex> vertices;
            std::vector<shared::Triangle> triangles;
            CUdeviceptr vertexBuffer = 0;
            CUdeviceptr triangleBuffer = 0;
            uint32_t vertexCount = 0;
            uint32_t triangleCount = 0;
        };
        std::vector<GeometryBuffers> geometryBuffers;
        
        // Light distribution data
        std::vector<float> lightProbabilities;
        CUdeviceptr lightDistBuffer = 0;
        
        // Mapping from RenderableNode to instance slot
        std::unordered_map<ItemID, uint32_t> nodeToInstanceSlot;
        std::unordered_map<ItemID, std::vector<uint32_t>> nodeToGeomInstSlots;
        
        // Stats
        uint32_t totalInstances = 0;
        uint32_t totalGeometryInstances = 0;
        uint32_t totalMaterials = 0;
        uint32_t totalVertices = 0;
        uint32_t totalTriangles = 0;
    };
    
    struct ConversionParams {
        bool enableInstancing = true;
        bool generateTangents = true;
        bool flipUVs = false;
        bool enableAreaLights = true;
        float defaultRoughness = 0.5f;
        float defaultMetallic = 0.0f;
        shared::RGB defaultBaseColor = shared::RGB(0.8f, 0.8f, 0.8f);
    };

public:
    RenderableNodeConverter(CUcontext cuContext);
    ~RenderableNodeConverter();
    
    // Main conversion function
    DeviceBuffers convertSceneGraph(
        const RenderableNode& rootNode,
        const ConversionParams& params = ConversionParams());
    
    // Update existing buffers with new transforms (for animation)
    void updateTransforms(
        DeviceBuffers& buffers,
        const RenderableNode& rootNode);
    
    // Free all allocated device memory
    void freeDeviceBuffers(DeviceBuffers& buffers);
    
private:
    CUcontext m_cuContext;
    
    // Material conversion
    uint32_t convertMaterial(
        const CgMaterial& cgMaterial,
        std::vector<shared::MaterialData>& materials,
        const ConversionParams& params);
    
    uint32_t convertMaterial(
        const Material& material,
        std::vector<shared::MaterialData>& materials,
        const ConversionParams& params);
    
    // Geometry conversion
    void convertCgModelToGeometry(
        const CgModel& cgModel,
        const Matrix4x4& transform,
        DeviceBuffers& buffers,
        std::vector<uint32_t>& geomInstSlots,
        const ConversionParams& params);
    
    // Instance conversion
    uint32_t createInstance(
        const RenderableNode& node,
        DeviceBuffers& buffers,
        const ConversionParams& params);
    
    // Recursive scene traversal
    void traverseSceneGraph(
        const RenderableNode& node,
        const Matrix4x4& parentTransform,
        DeviceBuffers& buffers,
        const ConversionParams& params);
    
    // Helper functions
    Matrix4x4 convertEigenToMatrix4x4(const Eigen::Affine3f& eigenTransform);
    Matrix3x3 extractNormalMatrix(const Matrix4x4& transform);
    float extractUniformScale(const Matrix4x4& transform);
    
    // Tangent space calculation
    void calculateTangents(
        std::vector<shared::Vertex>& vertices,
        const std::vector<shared::Triangle>& triangles);
    
    // Light distribution setup
    void setupLightDistribution(
        DeviceBuffers& buffers,
        const ConversionParams& params);
    
    // CUDA memory management
    CUdeviceptr allocateDeviceBuffer(size_t sizeInBytes);
    void copyToDevice(CUdeviceptr devicePtr, const void* hostPtr, size_t sizeInBytes);
    void freeDeviceBuffer(CUdeviceptr devicePtr);
    
    // Material texture loading helpers
    CUtexObject loadTexture(const std::string& path);
    shared::TexDimInfo getTextureDimensions(CUtexObject texObj);
    
    // Cache for shared geometry data (for instancing)
    struct GeometryCache {
        uint32_t refCount = 0;
        std::vector<uint32_t> geomInstSlots;
    };
    std::unordered_map<CgModel*, GeometryCache> m_geometryCache;
    
    // Cache for materials to avoid duplicates
    std::unordered_map<std::string, uint32_t> m_materialNameToSlot;
    
    // Texture cache
    std::unordered_map<std::string, CUtexObject> m_textureCache;
};

// Implementation file would go in RenderableNodeConverter.cpp