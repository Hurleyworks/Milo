#include "RenderableNodeConverter.h"
#include <cuda.h>
#include <g3log/g3log.hpp>
#include <algorithm>
#include <stack>

RenderableNodeConverter::RenderableNodeConverter(CUcontext cuContext)
    : m_cuContext(cuContext) {
    // Set CUDA context
    CUresult result = cuCtxSetCurrent(m_cuContext);
    if (result != CUDA_SUCCESS) {
        LOG(WARNING) << "Failed to set CUDA context in RenderableNodeConverter";
    }
}

RenderableNodeConverter::~RenderableNodeConverter() {
    // Clean up texture cache
    for (auto& [path, texObj] : m_textureCache) {
        if (texObj) {
            cuTexObjectDestroy(texObj);
        }
    }
}

RenderableNodeConverter::DeviceBuffers RenderableNodeConverter::convertSceneGraph(
    const RenderableNode& rootNode,
    const ConversionParams& params) {
    
    DeviceBuffers buffers;
    
    // Clear caches for new conversion
    m_geometryCache.clear();
    m_materialNameToSlot.clear();
    
    // Start recursive traversal from root
    Matrix4x4 identityTransform = Matrix4x4::Identity();
    traverseSceneGraph(rootNode, identityTransform, buffers, params);
    
    // Setup light distribution if we have emissive materials
    if (params.enableAreaLights) {
        setupLightDistribution(buffers, params);
    }
    
    // Allocate and upload device buffers
    
    // Materials
    if (!buffers.materials.empty()) {
        size_t materialBufferSize = buffers.materials.size() * sizeof(shared::MaterialData);
        buffers.materialBuffer = allocateDeviceBuffer(materialBufferSize);
        copyToDevice(buffers.materialBuffer, buffers.materials.data(), materialBufferSize);
        buffers.totalMaterials = buffers.materials.size();
    }
    
    // Geometry instances
    if (!buffers.geometryInstances.empty()) {
        size_t geomInstBufferSize = buffers.geometryInstances.size() * sizeof(shared::GeometryInstanceData);
        buffers.geometryInstanceBuffer = allocateDeviceBuffer(geomInstBufferSize);
        copyToDevice(buffers.geometryInstanceBuffer, buffers.geometryInstances.data(), geomInstBufferSize);
        buffers.totalGeometryInstances = buffers.geometryInstances.size();
    }
    
    // Instances
    if (!buffers.instances.empty()) {
        size_t instBufferSize = buffers.instances.size() * sizeof(shared::InstanceData);
        buffers.instanceBuffer = allocateDeviceBuffer(instBufferSize);
        copyToDevice(buffers.instanceBuffer, buffers.instances.data(), instBufferSize);
        buffers.totalInstances = buffers.instances.size();
    }
    
    // Geometry buffers (vertices and triangles)
    for (auto& geomBuf : buffers.geometryBuffers) {
        if (!geomBuf.vertices.empty()) {
            size_t vertexBufferSize = geomBuf.vertices.size() * sizeof(shared::Vertex);
            geomBuf.vertexBuffer = allocateDeviceBuffer(vertexBufferSize);
            copyToDevice(geomBuf.vertexBuffer, geomBuf.vertices.data(), vertexBufferSize);
            buffers.totalVertices += geomBuf.vertices.size();
        }
        
        if (!geomBuf.triangles.empty()) {
            size_t triangleBufferSize = geomBuf.triangles.size() * sizeof(shared::Triangle);
            geomBuf.triangleBuffer = allocateDeviceBuffer(triangleBufferSize);
            copyToDevice(geomBuf.triangleBuffer, geomBuf.triangles.data(), triangleBufferSize);
            buffers.totalTriangles += geomBuf.triangles.size();
        }
    }
    
    LOG(DBUG) << "Scene conversion complete:"
              << " Instances=" << buffers.totalInstances
              << " GeometryInstances=" << buffers.totalGeometryInstances
              << " Materials=" << buffers.totalMaterials
              << " Vertices=" << buffers.totalVertices
              << " Triangles=" << buffers.totalTriangles;
    
    return buffers;
}

void RenderableNodeConverter::traverseSceneGraph(
    const RenderableNode& node,
    const Matrix4x4& parentTransform,
    DeviceBuffers& buffers,
    const ConversionParams& params) {
    
    if (!node) return;
    
    // Get node's transform
    const SpaceTime& spacetime = node->getSpaceTime();
    Matrix4x4 localTransform = convertEigenToMatrix4x4(spacetime.worldTransform);
    Matrix4x4 worldTransform = parentTransform * localTransform;
    
    // Process this node if it has geometry
    CgModelPtr cgModel = node->getModel();
    if (cgModel && cgModel->isValid()) {
        // Create instance for this node
        uint32_t instanceSlot = createInstance(node, buffers, params);
        
        // Convert geometry if not already cached
        auto cacheIt = m_geometryCache.find(cgModel.get());
        if (cacheIt == m_geometryCache.end()) {
            // New geometry - convert it
            std::vector<uint32_t> geomInstSlots;
            convertCgModelToGeometry(*cgModel, worldTransform, buffers, geomInstSlots, params);
            
            // Cache the geometry instance slots
            GeometryCache cache;
            cache.refCount = 1;
            cache.geomInstSlots = geomInstSlots;
            m_geometryCache[cgModel.get()] = cache;
            
            // Store mapping
            buffers.nodeToGeomInstSlots[node->getID()] = geomInstSlots;
        } else {
            // Reuse existing geometry instances
            cacheIt->second.refCount++;
            buffers.nodeToGeomInstSlots[node->getID()] = cacheIt->second.geomInstSlots;
        }
        
        // Update instance data with proper transform
        if (instanceSlot < buffers.instances.size()) {
            shared::InstanceData& inst = buffers.instances[instanceSlot];
            inst.transform = worldTransform;
            inst.normalMatrix = extractNormalMatrix(worldTransform);
            inst.uniformScale = extractUniformScale(worldTransform);
            
            // Set geometry instance slots for this instance
            const auto& geomSlots = buffers.nodeToGeomInstSlots[node->getID()];
            // Note: In real implementation, would need to properly set up ROBuffer for geomInstSlots
        }
    }
    
    // Recursively process children
    for (const auto& [childId, child] : node->getChildren()) {
        traverseSceneGraph(child, worldTransform, buffers, params);
    }
}

void RenderableNodeConverter::convertCgModelToGeometry(
    const CgModel& cgModel,
    const Matrix4x4& transform,
    DeviceBuffers& buffers,
    std::vector<uint32_t>& geomInstSlots,
    const ConversionParams& params) {
    
    // Process each surface as a separate geometry instance
    for (const auto& surface : cgModel.S) {
        // Create new geometry buffers
        DeviceBuffers::GeometryBuffers geomBuf;
        
        // Collect unique vertex indices for this surface
        std::unordered_map<uint32_t, uint32_t> globalToLocalIndex;
        
        // First pass: identify unique vertices used by this surface
        for (int triIdx = 0; triIdx < surface.F.cols(); ++triIdx) {
            const Vector3u& tri = surface.F.col(triIdx);
            for (int i = 0; i < 3; ++i) {
                uint32_t globalIdx = tri[i];
                if (globalToLocalIndex.find(globalIdx) == globalToLocalIndex.end()) {
                    uint32_t localIdx = globalToLocalIndex.size();
                    globalToLocalIndex[globalIdx] = localIdx;
                }
            }
        }
        
        // Create vertex buffer for this surface
        geomBuf.vertices.resize(globalToLocalIndex.size());
        Matrix3x3 normalMatrix = extractNormalMatrix(transform);
        
        for (const auto& [globalIdx, localIdx] : globalToLocalIndex) {
            shared::Vertex& vertex = geomBuf.vertices[localIdx];
            
            // Transform position
            Vector3f pos = cgModel.V.col(globalIdx);
            Point3D worldPos = transform * Point3D(pos.x(), pos.y(), pos.z());
            vertex.position = worldPos;
            
            // Transform normal if available
            if (cgModel.N.cols() > globalIdx) {
                Vector3f norm = cgModel.N.col(globalIdx);
                Normal3D worldNorm = normalize(normalMatrix * Normal3D(norm.x(), norm.y(), norm.z()));
                vertex.normal = worldNorm;
            } else {
                // Default normal
                vertex.normal = Normal3D(0, 1, 0);
            }
            
            // Texture coordinates
            if (cgModel.UV0.cols() > globalIdx) {
                Vector2f uv = cgModel.UV0.col(globalIdx);
                vertex.texCoord = Point2D(uv.x(), params.flipUVs ? 1.0f - uv.y() : uv.y());
            } else {
                vertex.texCoord = Point2D(0, 0);
            }
            
            // Tangent will be calculated later if needed
            vertex.texCoord0Dir = Vector3D(1, 0, 0);
        }
        
        // Create triangle buffer with local indices
        geomBuf.triangles.reserve(surface.F.cols());
        for (int triIdx = 0; triIdx < surface.F.cols(); ++triIdx) {
            const Vector3u& globalTri = surface.F.col(triIdx);
            shared::Triangle localTri;
            localTri.index0 = globalToLocalIndex[globalTri[0]];
            localTri.index1 = globalToLocalIndex[globalTri[1]];
            localTri.index2 = globalToLocalIndex[globalTri[2]];
            geomBuf.triangles.push_back(localTri);
        }
        
        // Calculate tangents if requested
        if (params.generateTangents) {
            calculateTangents(geomBuf.vertices, geomBuf.triangles);
        }
        
        geomBuf.vertexCount = geomBuf.vertices.size();
        geomBuf.triangleCount = geomBuf.triangles.size();
        
        // Convert material
        uint32_t materialSlot = 0;
        if (surface.cgMaterial.isValid()) {
            materialSlot = convertMaterial(surface.cgMaterial, buffers.materials, params);
        } else {
            materialSlot = convertMaterial(surface.material, buffers.materials, params);
        }
        
        // Create geometry instance data
        shared::GeometryInstanceData geomInst;
        // Note: vertexBuffer and triangleBuffer pointers will be set after device allocation
        geomInst.materialSlot = materialSlot;
        geomInst.geomInstSlot = buffers.geometryInstances.size();
        
        // Add to buffers
        uint32_t geomBufIndex = buffers.geometryBuffers.size();
        buffers.geometryBuffers.push_back(std::move(geomBuf));
        buffers.geometryInstances.push_back(geomInst);
        geomInstSlots.push_back(geomInst.geomInstSlot);
    }
}

uint32_t RenderableNodeConverter::createInstance(
    const RenderableNode& node,
    DeviceBuffers& buffers,
    const ConversionParams& params) {
    
    shared::InstanceData instance;
    
    // Get transform from node
    const SpaceTime& spacetime = node->getSpaceTime();
    instance.transform = convertEigenToMatrix4x4(spacetime.worldTransform);
    instance.normalMatrix = extractNormalMatrix(instance.transform);
    instance.uniformScale = extractUniformScale(instance.transform);
    
    // For animation support - store previous transform
    // For now, use same as current
    instance.curToPrevTransform = Matrix4x4::Identity();
    
    uint32_t instanceSlot = buffers.instances.size();
    buffers.instances.push_back(instance);
    buffers.nodeToInstanceSlot[node->getID()] = instanceSlot;
    
    return instanceSlot;
}

uint32_t RenderableNodeConverter::convertMaterial(
    const CgMaterial& cgMaterial,
    std::vector<shared::MaterialData>& materials,
    const ConversionParams& params) {
    
    // Check if we've already converted this material
    std::string matName = cgMaterial.name.empty() ? "unnamed_mat" : cgMaterial.name;
    auto it = m_materialNameToSlot.find(matName);
    if (it != m_materialNameToSlot.end()) {
        return it->second;
    }
    
    shared::MaterialData matData;
    
    // Set material type - using SimplePBR as example
    matData.bsdfType = shared::BSDFType::SimplePBR;
    
    // Convert PBR properties
    // Base color
    if (cgMaterial.baseColorFactor.has_value()) {
        const auto& color = cgMaterial.baseColorFactor.value();
        // Note: would need to create constant texture or handle differently
    }
    
    // Load textures if available
    if (cgMaterial.baseColorTexture.has_value()) {
        // Load base color texture
        // matData.asSimplePBR.baseColor_opacity = loadTexture(textureInfo.path);
    }
    
    // Set default values for now
    matData.asSimplePBR.baseColor_opacity = 0; // null texture
    matData.asSimplePBR.occlusion_roughness_metallic = 0;
    
    // Handle emissive materials for area lights
    if (cgMaterial.emissiveFactor.has_value()) {
        const auto& emissive = cgMaterial.emissiveFactor.value();
        matData.emittance = RGB(emissive[0], emissive[1], emissive[2]);
    }
    
    uint32_t materialSlot = materials.size();
    materials.push_back(matData);
    m_materialNameToSlot[matName] = materialSlot;
    
    return materialSlot;
}

uint32_t RenderableNodeConverter::convertMaterial(
    const Material& material,
    std::vector<shared::MaterialData>& materials,
    const ConversionParams& params) {
    
    shared::MaterialData matData;
    
    // Simple material conversion - use defaults for now
    matData.bsdfType = shared::BSDFType::SimplePBR;
    matData.asSimplePBR.baseColor_opacity = 0;
    matData.asSimplePBR.occlusion_roughness_metallic = 0;
    
    uint32_t materialSlot = materials.size();
    materials.push_back(matData);
    
    return materialSlot;
}

void RenderableNodeConverter::calculateTangents(
    std::vector<shared::Vertex>& vertices,
    const std::vector<shared::Triangle>& triangles) {
    
    // Initialize tangent accumulation
    std::vector<Vector3D> tangents(vertices.size(), Vector3D(0, 0, 0));
    std::vector<Vector3D> bitangents(vertices.size(), Vector3D(0, 0, 0));
    
    // Calculate tangents per triangle
    for (const auto& tri : triangles) {
        const shared::Vertex& v0 = vertices[tri.index0];
        const shared::Vertex& v1 = vertices[tri.index1];
        const shared::Vertex& v2 = vertices[tri.index2];
        
        Vector3D e1 = v1.position - v0.position;
        Vector3D e2 = v2.position - v0.position;
        
        float du1 = v1.texCoord.x - v0.texCoord.x;
        float dv1 = v1.texCoord.y - v0.texCoord.y;
        float du2 = v2.texCoord.x - v0.texCoord.x;
        float dv2 = v2.texCoord.y - v0.texCoord.y;
        
        float det = du1 * dv2 - du2 * dv1;
        if (std::abs(det) < 1e-6f) {
            // Degenerate UV coordinates - use default tangent
            continue;
        }
        
        float invDet = 1.0f / det;
        Vector3D tangent = (e1 * dv2 - e2 * dv1) * invDet;
        Vector3D bitangent = (e2 * du1 - e1 * du2) * invDet;
        
        // Accumulate for vertices
        tangents[tri.index0] += tangent;
        tangents[tri.index1] += tangent;
        tangents[tri.index2] += tangent;
        
        bitangents[tri.index0] += bitangent;
        bitangents[tri.index1] += bitangent;
        bitangents[tri.index2] += bitangent;
    }
    
    // Orthogonalize and normalize tangents
    for (size_t i = 0; i < vertices.size(); ++i) {
        Vector3D& t = tangents[i];
        const Vector3D& b = bitangents[i];
        const Normal3D& n = vertices[i].normal;
        
        // Gram-Schmidt orthogonalization
        t = normalize(t - n * dot(n, t));
        
        // Calculate handedness
        float handedness = dot(cross(n, t), b) < 0.0f ? -1.0f : 1.0f;
        
        // Store tangent with handedness in w component (if we had w)
        vertices[i].texCoord0Dir = t;
    }
}

void RenderableNodeConverter::setupLightDistribution(
    DeviceBuffers& buffers,
    const ConversionParams& params) {
    
    // Calculate light probabilities based on emissive materials and geometry
    buffers.lightProbabilities.clear();
    
    for (size_t i = 0; i < buffers.geometryInstances.size(); ++i) {
        const auto& geomInst = buffers.geometryInstances[i];
        
        // Check if material is emissive
        if (geomInst.materialSlot < buffers.materials.size()) {
            const auto& mat = buffers.materials[geomInst.materialSlot];
            
            // Calculate emission power
            float power = mat.emittance.x + mat.emittance.y + mat.emittance.z;
            
            if (power > 0.0f) {
                // Weight by surface area (simplified - would need actual geometry area)
                float area = 1.0f; // Placeholder
                buffers.lightProbabilities.push_back(power * area);
            } else {
                buffers.lightProbabilities.push_back(0.0f);
            }
        }
    }
    
    // Normalize probabilities
    float totalPower = 0.0f;
    for (float prob : buffers.lightProbabilities) {
        totalPower += prob;
    }
    
    if (totalPower > 0.0f) {
        for (float& prob : buffers.lightProbabilities) {
            prob /= totalPower;
        }
    }
}

void RenderableNodeConverter::updateTransforms(
    DeviceBuffers& buffers,
    const RenderableNode& rootNode) {
    
    // Stack-based traversal for updating transforms
    std::stack<std::pair<RenderableNode, Matrix4x4>> nodeStack;
    nodeStack.push({rootNode, Matrix4x4::Identity()});
    
    while (!nodeStack.empty()) {
        auto [node, parentTransform] = nodeStack.top();
        nodeStack.pop();
        
        if (!node) continue;
        
        // Calculate world transform
        const SpaceTime& spacetime = node->getSpaceTime();
        Matrix4x4 localTransform = convertEigenToMatrix4x4(spacetime.worldTransform);
        Matrix4x4 worldTransform = parentTransform * localTransform;
        
        // Update instance transform if this node has one
        auto it = buffers.nodeToInstanceSlot.find(node->getID());
        if (it != buffers.nodeToInstanceSlot.end()) {
            uint32_t instanceSlot = it->second;
            if (instanceSlot < buffers.instances.size()) {
                shared::InstanceData& inst = buffers.instances[instanceSlot];
                
                // Store previous transform for motion vectors
                inst.curToPrevTransform = inst.transform;
                
                // Update current transform
                inst.transform = worldTransform;
                inst.normalMatrix = extractNormalMatrix(worldTransform);
                inst.uniformScale = extractUniformScale(worldTransform);
            }
        }
        
        // Add children to stack
        for (const auto& [childId, child] : node->getChildren()) {
            nodeStack.push({child, worldTransform});
        }
    }
    
    // Upload updated instance data to device
    if (!buffers.instances.empty() && buffers.instanceBuffer) {
        size_t bufferSize = buffers.instances.size() * sizeof(shared::InstanceData);
        copyToDevice(buffers.instanceBuffer, buffers.instances.data(), bufferSize);
    }
}

// Helper function implementations

Matrix4x4 RenderableNodeConverter::convertEigenToMatrix4x4(const Eigen::Affine3f& eigenTransform) {
    const auto& m = eigenTransform.matrix();
    return Matrix4x4(
        Vector4D(m(0,0), m(0,1), m(0,2), m(0,3)),
        Vector4D(m(1,0), m(1,1), m(1,2), m(1,3)),
        Vector4D(m(2,0), m(2,1), m(2,2), m(2,3)),
        Vector4D(m(3,0), m(3,1), m(3,2), m(3,3))
    );
}

Matrix3x3 RenderableNodeConverter::extractNormalMatrix(const Matrix4x4& transform) {
    // Extract 3x3 rotation/scale, invert, and transpose for normal transformation
    Matrix3x3 upper3x3(
        Vector3D(transform.c0.x, transform.c0.y, transform.c0.z),
        Vector3D(transform.c1.x, transform.c1.y, transform.c1.z),
        Vector3D(transform.c2.x, transform.c2.y, transform.c2.z)
    );
    return transpose(invert(upper3x3));
}

float RenderableNodeConverter::extractUniformScale(const Matrix4x4& transform) {
    // Extract uniform scale from transform matrix
    Vector3D scaleX(transform.c0.x, transform.c0.y, transform.c0.z);
    Vector3D scaleY(transform.c1.x, transform.c1.y, transform.c1.z);
    Vector3D scaleZ(transform.c2.x, transform.c2.y, transform.c2.z);
    
    float sx = length(scaleX);
    float sy = length(scaleY);
    float sz = length(scaleZ);
    
    // Return average scale (assumes relatively uniform scaling)
    return (sx + sy + sz) / 3.0f;
}

// CUDA memory management

CUdeviceptr RenderableNodeConverter::allocateDeviceBuffer(size_t sizeInBytes) {
    CUdeviceptr devicePtr = 0;
    CUresult result = cuMemAlloc(&devicePtr, sizeInBytes);
    if (result != CUDA_SUCCESS) {
        LOG(WARNING) << "Failed to allocate " << sizeInBytes << " bytes on device";
        return 0;
    }
    return devicePtr;
}

void RenderableNodeConverter::copyToDevice(CUdeviceptr devicePtr, const void* hostPtr, size_t sizeInBytes) {
    CUresult result = cuMemcpyHtoD(devicePtr, hostPtr, sizeInBytes);
    if (result != CUDA_SUCCESS) {
        LOG(WARNING) << "Failed to copy " << sizeInBytes << " bytes to device";
    }
}

void RenderableNodeConverter::freeDeviceBuffer(CUdeviceptr devicePtr) {
    if (devicePtr) {
        cuMemFree(devicePtr);
    }
}

void RenderableNodeConverter::freeDeviceBuffers(DeviceBuffers& buffers) {
    // Free material buffer
    freeDeviceBuffer(buffers.materialBuffer);
    
    // Free geometry instance buffer
    freeDeviceBuffer(buffers.geometryInstanceBuffer);
    
    // Free instance buffer
    freeDeviceBuffer(buffers.instanceBuffer);
    
    // Free geometry buffers
    for (auto& geomBuf : buffers.geometryBuffers) {
        freeDeviceBuffer(geomBuf.vertexBuffer);
        freeDeviceBuffer(geomBuf.triangleBuffer);
    }
    
    // Free light distribution buffer
    freeDeviceBuffer(buffers.lightDistBuffer);
    
    // Clear all data
    buffers = DeviceBuffers();
}

CUtexObject RenderableNodeConverter::loadTexture(const std::string& path) {
    // Check cache first
    auto it = m_textureCache.find(path);
    if (it != m_textureCache.end()) {
        return it->second;
    }
    
    // Load texture implementation would go here
    // This is a placeholder
    CUtexObject texObj = 0;
    
    // Cache the texture
    m_textureCache[path] = texObj;
    
    return texObj;
}

shared::TexDimInfo RenderableNodeConverter::getTextureDimensions(CUtexObject texObj) {
    shared::TexDimInfo dimInfo;
    // Implementation would query texture dimensions
    dimInfo.dimX = 1;
    dimInfo.dimY = 1;
    dimInfo.mipLevel = 0;
    return dimInfo;
}