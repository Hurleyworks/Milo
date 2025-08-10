#include "ShockerModel.h"

// =============================================================================
// ShockerModel Base Class Implementation
// =============================================================================

std::vector<uint32_t> ShockerModel::getGeomInstSlots() const
{
    std::vector<uint32_t> slots;
    for (const auto& surface : surfaces_) {
        slots.push_back(surface->geomInstSlot);
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
    
    for (const auto& surface : surfaces_) {
        combinedAABB_.minP.x = std::min(combinedAABB_.minP.x, surface->aabb.minP.x);
        combinedAABB_.minP.y = std::min(combinedAABB_.minP.y, surface->aabb.minP.y);
        combinedAABB_.minP.z = std::min(combinedAABB_.minP.z, surface->aabb.minP.z);
        
        combinedAABB_.maxP.x = std::max(combinedAABB_.maxP.x, surface->aabb.maxP.x);
        combinedAABB_.maxP.y = std::max(combinedAABB_.maxP.y, surface->aabb.maxP.y);
        combinedAABB_.maxP.z = std::max(combinedAABB_.maxP.z, surface->aabb.maxP.z);
    }
    
    // Handle empty case
    if (surfaces_.empty()) {
        combinedAABB_.minP = Point3D(0.0f, 0.0f, 0.0f);
        combinedAABB_.maxP = Point3D(0.0f, 0.0f, 0.0f);
    }
}

// =============================================================================
// ShockerTriangleModel Implementation
// =============================================================================

void ShockerTriangleModel::createFromRenderableNode(const RenderableNode& node, SlotFinder& slotFinder, RenderContext* renderContext)
{
    sourceNode_ = node.get();
    
    // Get the CgModel
    sabi::CgModelPtr model = node->getModel();
    if (!model) {
        LOG(WARNING) << "RenderableNode has no model: " << node->getName();
        return;
    }
    
    // Clear any existing geometry
    surfaces_.clear();
    
    // Create ShockerSurface for each surface
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
        
        // Create shocker surface
        auto surface = std::make_unique<shocker::ShockerSurface>();
        surface->geomInstSlot = slot;
        
        // Create appropriate geometry type based on surface properties
        createGeometryForSurface(model, surfIdx, surface.get(), renderContext);
        
        // Note: surface->mat (DisneyMaterial*) will be set later by MaterialHandler
        // Each surface gets its own DisneyMaterial since each ShockerSurface
        // can only have one material
        surface->mat = nullptr;
        
        // Store the surface
        surfaces_.push_back(std::move(surface));
    }
    
    // If model has particle data, create additional curve geometry instances
    if (hasParticleData) {
        uint32_t slot = slotFinder.getFirstAvailableSlot();
        if (slot != SlotFinder::InvalidSlotIndex) {
            slotFinder.setInUse(slot);
            
            auto surface = std::make_unique<shocker::ShockerSurface>();
            surface->geomInstSlot = slot;
            
            // Create curve geometry from particle data
            extractCurveGeometry(model, 0, surface.get());
            
            // Note: Curve geometry also needs a DisneyMaterial
            // Will be set by MaterialHandler based on curve rendering properties
            surface->mat = nullptr;
            
            surfaces_.push_back(std::move(surface));
        }
    }
    
    // Create ShockerSurfaceGroup containing all surfaces
    surfaceGroup_ = std::make_unique<shocker::ShockerSurfaceGroup>();
    for (const auto& surface : surfaces_) {
        surfaceGroup_->geomInsts.insert(surface.get());
    }
    
    // Calculate combined AABB
    calculateCombinedAABB();
    surfaceGroup_->aabb = combinedAABB_;
    
    // Set other geometry group properties
    surfaceGroup_->numEmitterPrimitives = 0;  // Will be set when materials are added
    surfaceGroup_->needsReallocation = 0;
    surfaceGroup_->needsRebuild = 1;  // Needs initial build
    surfaceGroup_->refittable = 0;    // Static geometry by default
    
    // Triangle model created successfully
}

void ShockerTriangleModel::createGeometryForSurface(
    const CgModelPtr& model,
    size_t surfaceIndex,
    shocker::ShockerSurface* surface,
    RenderContext* renderContext)
{
    // Determine appropriate geometry type for this surface
    if (shouldUseDisplacementGeometry(model, surfaceIndex)) {
        // TODO: Create TFDM or NRTDSM geometry
        // Surface would use displacement geometry (not yet implemented)
    }
    else if (shouldUseCurveGeometry(model, surfaceIndex)) {
        // Create curve geometry
        extractCurveGeometry(model, surfaceIndex, surface);
    }
    else {
        // Default to triangle geometry
        std::vector<shared::Vertex> vertices;
        std::vector<shared::Triangle> triangles;
        std::vector<uint32_t> materialIndices;
        
        extractTriangleGeometry(model, surfaceIndex, vertices, triangles, materialIndices);
        
        // Create triangle geometry
        TriangleGeometry triGeom;
        
        // If we have a render context and valid data, create GPU resources
        if (renderContext && !vertices.empty() && !triangles.empty()) {
            // Get CUDA context
            CUcontext cudaContext = renderContext->getCudaContext();
            optixu::Context optixContext = renderContext->getOptiXContext();
            
            // Create GPU buffers
            triGeom.vertexBuffer.initialize(cudaContext, cudau::BufferType::Device, vertices.size());
            triGeom.triangleBuffer.initialize(cudaContext, cudau::BufferType::Device, triangles.size());
            
            // Upload data to GPU
            triGeom.vertexBuffer.write(vertices.data(), vertices.size());
            triGeom.triangleBuffer.write(triangles.data(), triangles.size());
            
            // Note: OptiX geometry instance will be created later in ShockerSceneHandler
            // when building the acceleration structures
            
            LOG(DBUG) << "Created GPU geometry for surface " << surfaceIndex 
                      << " with " << vertices.size() << " vertices and " 
                      << triangles.size() << " triangles";
        } else if (!renderContext) {
            LOG(DBUG) << "No render context provided - GPU resources not allocated";
        }
        
        surface->geometry = std::move(triGeom);
        
        // Calculate AABB for this surface
        surface->aabb = calculateAABBForVertices(vertices);
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
    shocker::ShockerSurface* surface)
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
        
        // Created curve geometry with control points
    }
    
    surface->geometry = std::move(curveGeom);
    
    // Calculate AABB for curves
    surface->aabb.minP = Point3D(FLT_MAX, FLT_MAX, FLT_MAX);
    surface->aabb.maxP = Point3D(-FLT_MAX, -FLT_MAX, -FLT_MAX);
    
    for (const auto& p : model->P) {
        surface->aabb.minP.x = std::min(surface->aabb.minP.x, p.x());
        surface->aabb.minP.y = std::min(surface->aabb.minP.y, p.y());
        surface->aabb.minP.z = std::min(surface->aabb.minP.z, p.z());
        
        surface->aabb.maxP.x = std::max(surface->aabb.maxP.x, p.x());
        surface->aabb.maxP.y = std::max(surface->aabb.maxP.y, p.y());
        surface->aabb.maxP.z = std::max(surface->aabb.maxP.z, p.z());
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

void ShockerFlyweightModel::createFromRenderableNode(const RenderableNode& node, SlotFinder& slotFinder, RenderContext* renderContext)
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
    surfaceGroup_ = nullptr;  // We don't own it, just reference it
    
    // Copy the AABB from source
    combinedAABB_ = sourceModel_->getAABB();
    
    // Flyweight model created referencing source
}

// =============================================================================
// ShockerPhantomModel Implementation
// =============================================================================

void ShockerPhantomModel::createFromRenderableNode(const RenderableNode& node, SlotFinder& slotFinder, RenderContext* renderContext)
{
    sourceNode_ = node.get();
    
    // Phantom models have no visible geometry
    // They're used for physics/collision only
    
    // Create a single empty geometry instance for the instance system
    uint32_t slot = slotFinder.getFirstAvailableSlot();
    if (slot != SlotFinder::InvalidSlotIndex) {
        slotFinder.setInUse(slot);
        
        auto surface = std::make_unique<shocker::ShockerSurface>();
        surface->geomInstSlot = slot;
        surface->mat = nullptr;  // No material for phantom
        
        // Set empty AABB for the surface
        surface->aabb.minP = Point3D(0.0f, 0.0f, 0.0f);
        surface->aabb.maxP = Point3D(0.0f, 0.0f, 0.0f);
        
        // Empty triangle geometry (no actual triangles)
        TriangleGeometry emptyGeom;
        surface->geometry = std::move(emptyGeom);
        
        surfaces_.push_back(std::move(surface));
    }
    
    // Create empty surface group
    surfaceGroup_ = std::make_unique<shocker::ShockerSurfaceGroup>();
    if (!surfaces_.empty()) {
        surfaceGroup_->geomInsts.insert(surfaces_[0].get());
    }
    surfaceGroup_->numEmitterPrimitives = 0;
    surfaceGroup_->needsReallocation = 0;
    surfaceGroup_->needsRebuild = 0;
    surfaceGroup_->refittable = 0;
    
    // Set empty AABB
    combinedAABB_.minP = Point3D(0.0f, 0.0f, 0.0f);
    combinedAABB_.maxP = Point3D(0.0f, 0.0f, 0.0f);
    surfaceGroup_->aabb = combinedAABB_;
    
    // Phantom model created (no logging needed for routine operations)
}