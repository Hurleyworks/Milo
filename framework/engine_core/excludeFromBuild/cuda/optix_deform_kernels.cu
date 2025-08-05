#include "../milo_shared.h"

using namespace milo_shared;
using namespace shared;

// Resets deformed mesh to original vertex positions
// This is called when transitioning from deformed to undeformed state
CUDA_DEVICE_KERNEL void resetDeform (
    const Vertex* originalVertices,
    Vertex* currentVertices,
    uint32_t numVertices)
{
    uint32_t vIdx = blockDim.x * blockIdx.x + threadIdx.x;
    if (vIdx >= numVertices)
        return;

    // Copy original position back to current vertex buffer
    currentVertices[vIdx].position = originalVertices[vIdx].position;

    // Clear normal for recomputation
    currentVertices[vIdx].normal = Vector3D_T<float, true>();
}

CUDA_DEVICE_KERNEL void deform (
    const Vertex* originalVertices,
    Vertex* vertices,
    uint32_t numVertices,
    const float3* deformedPositions)
{
    uint32_t vIdx = blockDim.x * blockIdx.x + threadIdx.x;
    if (vIdx >= numVertices)
        return;

    // Just copy the deformed position directly
    vertices[vIdx].position = Point3D_T<float> (
        deformedPositions[vIdx].x,
        deformedPositions[vIdx].y,
        deformedPositions[vIdx].z);

    // Clear normal for recomputation
    vertices[vIdx].normal = Vector3D_T<float, true>();
}

// Computes and accumulates vertex normals for each triangle face
// Uses atomic operations to safely accumulate when vertices are shared
CUDA_DEVICE_KERNEL void accumulateVertexNormals (
    Vertex* vertices,
    Triangle* triangles,
    uint32_t numTriangles)
{
    uint32_t triIdx = blockDim.x * blockIdx.x + threadIdx.x;
    if (triIdx >= numTriangles)
        return;

    const Triangle& tri = triangles[triIdx];
    Vertex& v0 = vertices[tri.index0];
    Vertex& v1 = vertices[tri.index1];
    Vertex& v2 = vertices[tri.index2];

    const auto atomicAddNormalAsInt32 = [] (Vector3D_T<float, true>* dstN, const int3& vn)
    {
        atomicAdd (reinterpret_cast<int32_t*> (&dstN->x), vn.x);
        atomicAdd (reinterpret_cast<int32_t*> (&dstN->y), vn.y);
        atomicAdd (reinterpret_cast<int32_t*> (&dstN->z), vn.z);
    };

    // Compute face normal using cross product of triangle edges
    auto edge1 = Vector3D_T<float, false> (v1.position - v0.position);
    auto edge2 = Vector3D_T<float, false> (v2.position - v0.position);
    auto normal = cross (edge1, edge2);
    Vector3D_T<float, true> vn = Vector3D_T<float, true> (normalize (normal));

    // Convert to fixed point for atomic operations
    constexpr int32_t coeffFloatToFixed = 1 << 24;
    int32_t vnx = static_cast<int32_t> (vn.x * coeffFloatToFixed);
    int32_t vny = static_cast<int32_t> (vn.y * coeffFloatToFixed);
    int32_t vnz = static_cast<int32_t> (vn.z * coeffFloatToFixed);
    int3 vnInt32 = make_int3 (vnx, vny, vnz);

    // Accumulate to all vertices of this triangle
    atomicAddNormalAsInt32 (&v0.normal, vnInt32);
    atomicAddNormalAsInt32 (&v1.normal, vnInt32);
    atomicAddNormalAsInt32 (&v2.normal, vnInt32);
}

// Normalizes the accumulated vertex normals after accumulation is complete
CUDA_DEVICE_KERNEL void normalizeVertexNormals (
    Vertex* vertices,
    uint32_t numVertices)
{
    uint32_t vIdx = blockDim.x * blockIdx.x + threadIdx.x;
    if (vIdx >= numVertices)
        return;

    // Convert from fixed point back to floating point
    Vector3D_T<float, true> vn = vertices[vIdx].normal;
    int32_t vnx = *reinterpret_cast<int32_t*> (&vn.x);
    int32_t vny = *reinterpret_cast<int32_t*> (&vn.y);
    int32_t vnz = *reinterpret_cast<int32_t*> (&vn.z);

    constexpr float coeffFixedToFloat = 1.0f / (1 << 24);
    vn = Vector3D_T<float, true> (
        vnx * coeffFixedToFloat,
        vny * coeffFixedToFloat,
        vnz * coeffFixedToFloat);

    // Normalize and store final normal
    vertices[vIdx].normal = normalize (vn);
}