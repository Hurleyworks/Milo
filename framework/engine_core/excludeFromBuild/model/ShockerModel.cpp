#include "ShockerModel.h"

// =============================================================================
// ShockerModel Base Class Implementation
// =============================================================================

std::vector<uint32_t> ShockerModel::getGeomInstSlots() const
{
    std::vector<uint32_t> slots;
    for (const auto& geomInst : geometryInstances_) {
        slots.push_back(geomInst->geomInstSlot);
    }
    return slots;
}

Matrix4x4 ShockerModel::convertSpaceTimeToMatrix(const SpaceTime& st)
{
    const Eigen::Matrix4f& eigenMat = st.worldTransform.matrix();
    
    Matrix4x4 mat;
    // Convert from Eigen column-major to our column-major format
    mat.c0.x = eigenMat(0, 0);
    mat.c0.y = eigenMat(1, 0);
    mat.c0.z = eigenMat(2, 0);
    mat.c0.w = eigenMat(3, 0);
    
    mat.c1.x = eigenMat(0, 1);
    mat.c1.y = eigenMat(1, 1);
    mat.c1.z = eigenMat(2, 1);
    mat.c1.w = eigenMat(3, 1);
    
    mat.c2.x = eigenMat(0, 2);
    mat.c2.y = eigenMat(1, 2);
    mat.c2.z = eigenMat(2, 2);
    mat.c2.w = eigenMat(3, 2);
    
    mat.c3.x = eigenMat(0, 3);
    mat.c3.y = eigenMat(1, 3);
    mat.c3.z = eigenMat(2, 3);
    mat.c3.w = eigenMat(3, 3);
    
    return mat;
}

Matrix3x3 ShockerModel::calculateNormalMatrix(const Matrix4x4& transform)
{
    // Extract upper-left 3x3
    Matrix3x3 upperLeft;
    upperLeft.m00 = transform.c0.x;
    upperLeft.m01 = transform.c1.x;
    upperLeft.m02 = transform.c2.x;
    upperLeft.m10 = transform.c0.y;
    upperLeft.m11 = transform.c1.y;
    upperLeft.m12 = transform.c2.y;
    upperLeft.m20 = transform.c0.z;
    upperLeft.m21 = transform.c1.z;
    upperLeft.m22 = transform.c2.z;
    
    // Normal matrix is inverse transpose of upper-left 3x3
    return transpose(invert(upperLeft));
}

void ShockerModel::calculateCombinedAABB()
{
    combinedAABB_.minP = Point3D(FLT_MAX, FLT_MAX, FLT_MAX);
    combinedAABB_.maxP = Point3D(-FLT_MAX, -FLT_MAX, -FLT_MAX);
    
    for (const auto& geomInst : geometryInstances_) {
        combinedAABB_.minP.x = std::min(combinedAABB_.minP.x, geomInst->aabb.minP.x);
        combinedAABB_.minP.y = std::min(combinedAABB_.minP.y, geomInst->aabb.minP.y);
        combinedAABB_.minP.z = std::min(combinedAABB_.minP.z, geomInst->aabb.minP.z);
        
        combinedAABB_.maxP.x = std::max(combinedAABB_.maxP.x, geomInst->aabb.maxP.x);
        combinedAABB_.maxP.y = std::max(combinedAABB_.maxP.y, geomInst->aabb.maxP.y);
        combinedAABB_.maxP.z = std::max(combinedAABB_.maxP.z, geomInst->aabb.maxP.z);
    }
    
    // Handle empty case
    if (geometryInstances_.empty()) {
        combinedAABB_.minP = Point3D(0.0f, 0.0f, 0.0f);
        combinedAABB_.maxP = Point3D(0.0f, 0.0f, 0.0f);
    }
}

// =============================================================================
// ShockerTriangleModel Implementation
// =============================================================================

void ShockerTriangleModel::createFromRenderableNode(const RenderableNode& node, SlotFinder& slotFinder)
{
    sourceNode_ = node.get();
    
    // Get the CgModel
    sabi::CgModelPtr model = node->getModel();
    if (!model) {
        LOG(WARNING) << "RenderableNode has no model: " << node->getName();
        return;
    }
    
    // Clear any existing geometry
    geometryInstances_.clear();
    
    // Create GeometryInstance for each surface
    size_t numSurfaces = model->S.size();
    if (numSurfaces == 0) {
        LOG(WARNING) << "Model has no surfaces: " << node->getName();
        return;
    }
    
    // Check if model has particle/curve data (applies to whole model, not per-surface)
    bool hasParticleData = model->P.size() > 0;
    
    // Process each surface
    for (size_t surfIdx = 0; surfIdx < numSurfaces; ++surfIdx) {
        // Allocate slot for this geometry instance
        uint32_t slot = slotFinder.getFirstAvailableSlot();
        if (slot == SlotFinder::InvalidSlotIndex) {
            LOG(WARNING) << "Failed to allocate geometry instance slot for surface " << surfIdx;
            continue;
        }
        slotFinder.setInUse(slot);
        
        // Create geometry instance
        auto geomInst = std::make_unique<GeometryInstance>();
        geomInst->geomInstSlot = slot;
        
        // Create appropriate geometry type based on surface properties
        createGeometryForSurface(model, surfIdx, geomInst.get());
        
        // Note: geomInst->mat (DisneyMaterial*) will be set later by MaterialHandler
        // Each surface gets its own DisneyMaterial since each GeometryInstance
        // can only have one material
        geomInst->mat = nullptr;
        
        // Store the geometry instance
        geometryInstances_.push_back(std::move(geomInst));
    }
    
    // If model has particle data, create additional curve geometry instances
    if (hasParticleData) {
        uint32_t slot = slotFinder.getFirstAvailableSlot();
        if (slot != SlotFinder::InvalidSlotIndex) {
            slotFinder.setInUse(slot);
            
            auto geomInst = std::make_unique<GeometryInstance>();
            geomInst->geomInstSlot = slot;
            
            // Create curve geometry from particle data
            extractCurveGeometry(model, 0, geomInst.get());
            
            // Note: Curve geometry also needs a DisneyMaterial
            // Will be set by MaterialHandler based on curve rendering properties
            geomInst->mat = nullptr;
            
            geometryInstances_.push_back(std::move(geomInst));
        }
    }
    
    // Create GeometryGroup containing all instances
    geometryGroup_ = std::make_unique<GeometryGroup>();
    for (const auto& geomInst : geometryInstances_) {
        geometryGroup_->geomInsts.insert(geomInst.get());
    }
    
    // Calculate combined AABB
    calculateCombinedAABB();
    geometryGroup_->aabb = combinedAABB_;
    
    // Set other geometry group properties
    geometryGroup_->numEmitterPrimitives = 0;  // Will be set when materials are added
    geometryGroup_->needsReallocation = 0;
    geometryGroup_->needsRebuild = 1;  // Needs initial build
    geometryGroup_->refittable = 0;    // Static geometry by default
    
    LOG(DBUG) << "Created ShockerTriangleModel with " << geometryInstances_.size() 
              << " geometry instances for " << node->getName();
}

void ShockerTriangleModel::createGeometryForSurface(
    const CgModelPtr& model,
    size_t surfaceIndex,
    GeometryInstance* geomInst)
{
    // Determine appropriate geometry type for this surface
    if (shouldUseDisplacementGeometry(model, surfaceIndex)) {
        // TODO: Create TFDM or NRTDSM geometry
        LOG(DBUG) << "Surface " << surfaceIndex << " would use displacement geometry (not yet implemented)";
    }
    else if (shouldUseCurveGeometry(model, surfaceIndex)) {
        // Create curve geometry
        extractCurveGeometry(model, surfaceIndex, geomInst);
    }
    else {
        // Default to triangle geometry
        std::vector<shared::Vertex> vertices;
        std::vector<shared::Triangle> triangles;
        std::vector<uint32_t> materialIndices;
        
        extractTriangleGeometry(model, surfaceIndex, vertices, triangles, materialIndices);
        
        // Create triangle geometry
        // Note: TriangleGeometry uses cudau::TypedBuffer, not std::vector
        // For now, just create empty buffers - actual GPU allocation would happen later
        TriangleGeometry triGeom;
        // triGeom.vertexBuffer will be allocated and filled with vertices data
        // triGeom.triangleBuffer will be allocated and filled with triangles data
        // Note: TriangleGeometry doesn't have materialIndexBuffer - each GeometryInstance has one material
        
        geomInst->geometry = std::move(triGeom);
        
        // Calculate AABB for this surface
        geomInst->aabb = calculateAABBForVertices(vertices);
    }
}

bool ShockerTriangleModel::shouldUseCurveGeometry(const CgModelPtr& model, size_t surfaceIndex) const
{
    // Check if this surface should be rendered as curves (e.g., hair, fur)
    // This could be based on surface tags, material properties, etc.
    if (surfaceIndex < model->S.size()) {
        const auto& surface = model->S[surfaceIndex];
        // Check for curve-related tags or properties
        // For now, return false - will be extended later
    }
    return false;
}

bool ShockerTriangleModel::shouldUseDisplacementGeometry(const CgModelPtr& model, size_t surfaceIndex) const
{
    // Check if this surface has displacement data
    if (model->VD.cols() > 0 && surfaceIndex < model->S.size()) {
        // Has displacement vectors - could use TFDM or NRTDSM
        return true;
    }
    return false;
}

void ShockerTriangleModel::extractCurveGeometry(
    const CgModelPtr& model,
    size_t surfaceIndex,
    GeometryInstance* geomInst)
{
    CurveGeometry curveGeom;
    
    // Extract curve data from particle data or special surface properties
    if (model->P.size() > 0) {
        // Convert particle data to curves
        // Each particle becomes a curve control point
        std::vector<Point3D> controlPoints;
        for (const auto& particle : model->P) {
            controlPoints.emplace_back(particle.x(), particle.y(), particle.z());
        }
        
        // In real implementation, would create proper curve segments
        // For now, just store the control points
        
        LOG(DBUG) << "Created curve geometry with " << controlPoints.size() << " control points";
    }
    
    geomInst->geometry = std::move(curveGeom);
    
    // Calculate AABB for curves
    geomInst->aabb.minP = Point3D(FLT_MAX, FLT_MAX, FLT_MAX);
    geomInst->aabb.maxP = Point3D(-FLT_MAX, -FLT_MAX, -FLT_MAX);
    
    for (const auto& p : model->P) {
        geomInst->aabb.minP.x = std::min(geomInst->aabb.minP.x, p.x());
        geomInst->aabb.minP.y = std::min(geomInst->aabb.minP.y, p.y());
        geomInst->aabb.minP.z = std::min(geomInst->aabb.minP.z, p.z());
        
        geomInst->aabb.maxP.x = std::max(geomInst->aabb.maxP.x, p.x());
        geomInst->aabb.maxP.y = std::max(geomInst->aabb.maxP.y, p.y());
        geomInst->aabb.maxP.z = std::max(geomInst->aabb.maxP.z, p.z());
    }
}

void ShockerTriangleModel::extractTriangleGeometry(
    const CgModelPtr& model,
    size_t surfaceIndex,
    std::vector<shared::Vertex>& vertices,
    std::vector<shared::Triangle>& triangles,
    std::vector<uint32_t>& materialIndices)
{
    // Get surface
    if (surfaceIndex >= model->S.size()) {
        LOG(WARNING) << "Invalid surface index: " << surfaceIndex;
        return;
    }
    
    const auto& surface = model->S[surfaceIndex];
    
    // Each surface has its own F (face/triangle indices)
    const MatrixXu& F = surface.F;
    
    // Build vertex buffer (may have duplicates for different surfaces)
    std::unordered_map<uint32_t, uint32_t> globalToLocalVertex;
    
    for (int triIdx = 0; triIdx < F.cols(); ++triIdx) {
        auto tri = F.col(triIdx);
        
        // Process each vertex of the triangle
        uint32_t localIndices[3];
        for (int v = 0; v < 3; ++v) {
            uint32_t globalIdx = tri(v);
            
            // Check if we've already added this vertex
            auto it = globalToLocalVertex.find(globalIdx);
            if (it != globalToLocalVertex.end()) {
                localIndices[v] = it->second;
            } else {
                // Add new vertex
                shared::Vertex vertex;
                
                // Position
                if (globalIdx < model->V.cols()) {
                    auto p = model->V.col(globalIdx);
                    vertex.position = Point3D(p.x(), p.y(), p.z());
                }
                
                // Normal
                if (globalIdx < model->N.cols()) {
                    auto n = model->N.col(globalIdx);
                    vertex.normal = Normal3D(n.x(), n.y(), n.z());
                }
                
                // Texture coordinates
                if (model->UV0.cols() > globalIdx) {
                    auto uv = model->UV0.col(globalIdx);
                    vertex.texCoord = Point2D(uv.x(), uv.y());
                } else {
                    vertex.texCoord = Point2D(0.0f, 0.0f);
                }
                
                // texCoord0Dir (tangent vector for texture mapping)
                // Would need to compute from UV derivatives - for now use default
                vertex.texCoord0Dir = Vector3D(1.0f, 0.0f, 0.0f);  // Default tangent
                
                uint32_t localIdx = static_cast<uint32_t>(vertices.size());
                vertices.push_back(vertex);
                globalToLocalVertex[globalIdx] = localIdx;
                localIndices[v] = localIdx;
            }
        }
        
        // Create triangle with local indices
        triangles.emplace_back(localIndices[0], localIndices[1], localIndices[2]);
        
        // Material index (same for all triangles in a surface)
        materialIndices.push_back(static_cast<uint32_t>(surfaceIndex));
    }
}

AABB ShockerTriangleModel::calculateAABBForVertices(const std::vector<shared::Vertex>& vertices)
{
    AABB aabb;
    aabb.minP = Point3D(FLT_MAX, FLT_MAX, FLT_MAX);
    aabb.maxP = Point3D(-FLT_MAX, -FLT_MAX, -FLT_MAX);
    
    for (const auto& vertex : vertices) {
        aabb.minP.x = std::min(aabb.minP.x, vertex.position.x);
        aabb.minP.y = std::min(aabb.minP.y, vertex.position.y);
        aabb.minP.z = std::min(aabb.minP.z, vertex.position.z);
        
        aabb.maxP.x = std::max(aabb.maxP.x, vertex.position.x);
        aabb.maxP.y = std::max(aabb.maxP.y, vertex.position.y);
        aabb.maxP.z = std::max(aabb.maxP.z, vertex.position.z);
    }
    
    // Handle empty case
    if (vertices.empty()) {
        aabb.minP = Point3D(0.0f, 0.0f, 0.0f);
        aabb.maxP = Point3D(0.0f, 0.0f, 0.0f);
    }
    
    return aabb;
}

// =============================================================================
// ShockerFlyweightModel Implementation
// =============================================================================

void ShockerFlyweightModel::createFromRenderableNode(const RenderableNode& node, SlotFinder& slotFinder)
{
    sourceNode_ = node.get();
    
    // Flyweight models don't create geometry - they reference another model's geometry
    // The geometry group will be set to point to the source model's geometry group
    
    if (!sourceModel_) {
        LOG(WARNING) << "Flyweight model has no source model set: " << node->getName();
        // Set empty AABB when no source
        combinedAABB_.minP = Point3D(0.0f, 0.0f, 0.0f);
        combinedAABB_.maxP = Point3D(0.0f, 0.0f, 0.0f);
        return;
    }
    
    // Reference the source model's geometry group
    geometryGroup_ = nullptr;  // We don't own it, just reference it
    
    // Copy the AABB from source
    combinedAABB_ = sourceModel_->getAABB();
    
    LOG(DBUG) << "Created ShockerFlyweightModel referencing source for " << node->getName();
}

// =============================================================================
// ShockerPhantomModel Implementation
// =============================================================================

void ShockerPhantomModel::createFromRenderableNode(const RenderableNode& node, SlotFinder& slotFinder)
{
    sourceNode_ = node.get();
    
    // Phantom models have no visible geometry
    // They're used for physics/collision only
    
    // Create empty geometry group
    geometryGroup_ = std::make_unique<GeometryGroup>();
    geometryGroup_->numEmitterPrimitives = 0;
    geometryGroup_->needsReallocation = 0;
    geometryGroup_->needsRebuild = 0;
    geometryGroup_->refittable = 0;
    
    // Set empty AABB
    combinedAABB_.minP = Point3D(0.0f, 0.0f, 0.0f);
    combinedAABB_.maxP = Point3D(0.0f, 0.0f, 0.0f);
    geometryGroup_->aabb = combinedAABB_;
    
    LOG(DBUG) << "Created ShockerPhantomModel for " << node->getName();
}