#pragma once

using Eigen::Vector3f;

// a Surface is a group of triangles with a unique Material
struct CgModelSurface
{
    std::string name;
    MatrixXu F; // triangle indices
    Material material;
    CgMaterial cgMaterial;
    float maxSmoothingAngle = 0.0f; // Maximum smoothing angle in radians

  
    bool materialHasChanged = false;

    const size_t triangleCount() const { return F.cols(); }
    MatrixXu& indices() { return F; }
    const MatrixXu& indices() const { return F; }

    uint32_t vertexCount = 0;

    float computeSurfaceArea (const MatrixXf& V) const
    {
        float area = 0.0f;

        for (int i = 0; i < F.cols(); ++i)
        {
            const Vector3u& tri = F.col (i);
            const Vector3f& v0 = V.col (tri[0]);
            const Vector3f& v1 = V.col (tri[1]);
            const Vector3f& v2 = V.col (tri[2]);

            Vector3f e1 = v1 - v0;
            Vector3f e2 = v2 - v0;

            float triArea = 0.5f * e1.cross (e2).norm();

            // Check for degenerate triangles
            if (std::isnan (triArea) || std::isinf (triArea) || triArea <= std::numeric_limits<float>::epsilon())
            {
                // Skip degenerate triangles and log a warning
                LOG (CRITICAL) << "Warning: Degenerate triangle found in surface. Skipping...";
                continue;
            }

            area += triArea;
        }

        return area;
    }

    Vector3f computeCentroid (const MatrixXf& V) const
    {
        Vector3f centroid = Vector3f::Zero();

        for (int i = 0; i < F.cols(); ++i)
        {
            const Vector3u& tri = F.col (i);
            const Vector3f& v0 = V.col (tri[0]);
            const Vector3f& v1 = V.col (tri[1]);
            const Vector3f& v2 = V.col (tri[2]);

            centroid += (v0 + v1 + v2) / 3.0f;
        }

        centroid /= static_cast<float> (F.cols());

        return centroid;
    }
};