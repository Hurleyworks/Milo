#pragma once

#include "CgModelSurface.h"

using CgModelPtr = std::shared_ptr<struct CgModel>;
using Eigen::Vector3f;

struct CgModel
{
    static CgModelPtr create() { return std::make_shared<CgModel>(); }

    MatrixXf V;   // vertices
    MatrixXf VD;  // displaced vertices
    MatrixXf N;   // vertex normals
    MatrixXf FN;  // face normals
    ParticleData P; // particle data
    MatrixXf UV0; // uv0
    MatrixXf UV1; // uv1

    // list of Surfaces and surface attributes
    std::vector<CgModelSurface> S;
    std::vector<Texture> textures;
    std::vector<Image> images;
    std::vector<Sampler> samplers;

    
    // CgMaterial Texture/Image/Sampler support
    std::vector<sabi::CgTexture> cgTextures;
    std::vector<sabi::CgImage> cgImages;
    std::vector<sabi::CgSampler> cgSamplers;
    

    fs::path contentDirectory;

    size_t triCount = 0; // must be computed
    size_t vertexCount() const { return V.cols(); }
    size_t triangleCount() 
    {
        // compute total face count if neccessary
        if (triCount == 0 && S.size())
        {
            MatrixXu allIndices;
            getAllSurfaceIndices (allIndices);
        }
        return triCount;
    }

    void reset()
    {
        V.resize (3, 0);
        N.resize (3, 0);
        UV0.resize (2, 0);
        UV1.resize (2, 0);
        triCount = 0;
        S.clear();
        textures.clear();
        images.clear();
        samplers.clear();
    }

    Eigen::AlignedBox3f computeBoundingBox() const
    {
        if (V.cols() == 0)
        {
            return Eigen::AlignedBox3f();
        }

        Eigen::AlignedBox3f bbox;
        bbox.min() = V.rowwise().minCoeff();
        bbox.max() = V.rowwise().maxCoeff();

        return bbox;
    }

     float computeSurfaceArea() const
    {
        float totalArea = 0.0f;

        for (const auto& surface : S)
        {
            const MatrixXu& F = surface.indices();

            for (int i = 0; i < F.cols(); ++i)
            {
                const Vector3u& tri = F.col (i);
                const Vector3f& v0 = V.col (tri[0]);
                const Vector3f& v1 = V.col (tri[1]);
                const Vector3f& v2 = V.col (tri[2]);

                Vector3f e1 = v1 - v0;
                Vector3f e2 = v2 - v0;

                float area = 0.5f * e1.cross (e2).norm();

                // Check for degenerate triangles
                if (std::isnan (area) || std::isinf (area) || area <= std::numeric_limits<float>::epsilon())
                {
                    // Skip degenerate triangles and log a warning
                    LOG (CRITICAL) << "Warning: Degenerate triangle found. Skipping...";
                    continue;
                }

                totalArea += area;
            }
        }

        return totalArea;
    }

    bool isValid()
    {
        if (V.cols() < 3 || N.cols() < 3) return false;
        if (N.cols() > 0 && V.cols() != N.cols()) return false;
        if (triangleCount() == 0) return false;
        if (S.size() == 0) return false;
        for (const auto& s : S)
            if (s.vertexCount == 0) return false;

        return true;
    }

    void transformVertices (const Eigen::Affine3f& t)
    {
        for (int i = 0; i < V.cols(); ++i)
        {
            Eigen::Vector3f p = V.col (i);
            V.col (i) = t * p;
        }
    }

    void getAllSurfaceIndices (MatrixXu& allIndices, bool unwelded = false) 
    {
        triCount = 0;
        for (const auto& s : S)
        {
            triCount += s.triangleCount();
        }

        allIndices.resize (3, triCount);

        int index = 0;
        for (const auto& s : S)
        {
            size_t triCount = s.triangleCount();

            for (int i = 0; i < triCount; i++)
                allIndices.col (index++) = s.indices().col (i);
        }
    }
    void debugMaterials()
    {
        for (auto& surface : S)
        {
           //  auto& mat = surface.material;
           // mat.debug();
        }

    }
};

using CgModelList = std::vector<CgModelPtr>;