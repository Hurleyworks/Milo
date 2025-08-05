#pragma once

#include "../milo_shared.h"
#include "../handlers/Handlers.h"

using Eigen::Vector3f;

// Function to populate OptiX vertex structs from a cgModel
inline std::vector<shared::Vertex> populateVertices (const sabi::CgModelPtr& model)
{
   // LOG (DBUG) << _FN_;

    assert (model && model->V.cols() > 0);

    std::vector<shared::Vertex> vertices (model->V.cols());

    const auto makeCoordinateSystem = [] (const Normal3D& normal, Vector3D* tangent, Vector3D* bitangent)
    {
        float sign = normal.z >= 0 ? 1.0f : -1.0f;
        const float a = -1 / (sign + normal.z);
        const float b = normal.x * normal.y * a;
        *tangent = Vector3D (1 + sign * normal.x * normal.x * a, sign * b, -sign * normal.x);
        *bitangent = Vector3D (b, sign + normal.y * normal.y * a, -normal.y);
    };

    for (int i = 0; i < model->V.cols(); ++i)
    {
        shared::Vertex vertex;

        Eigen::Vector3f p = model->V.col (i);
       // LOG (DBUG) << p.x() << ", " << p.y() << ", " << p.z();
        vertex.position = Point3D (p.x(), p.y(), p.z());

        Eigen::Vector3f n = model->N.col (i);
        for (int j = 0; j < 3; j++)
        {
            if (!std::isfinite (n[j]))
            {
                // FIXME how to handle this properly????
                n = Eigen::Vector3f::UnitY();
                // LOG (CRITICAL) << "This normal is NAN";
            }
        }
        vertex.normal = Normal3D (n.x(), n.y(), n.z());

        Vector3D tangent, bitangent;
        makeCoordinateSystem (vertex.normal, &tangent, &bitangent);

        if (model->UV0.cols())
        {
            Eigen::Vector2f uv = model->UV0.col (i);
            vertex.texCoord = Point2D (uv.x(), uv.y());
            vertex.texCoord0Dir = normalize (Vector3D (tangent.x, tangent.y, tangent.z));
        }
        else
        {
            vertex.texCoord = Point2D (0.0f, 0.0f);
            vertex.texCoord0Dir = normalize (Vector3D (0.0f, 0.0f, 0.0f));
        }

        // LOG (DBUG) << "TexCoord0Dir " << vertex.texCoord0Dir.x << ", " << vertex.texCoord0Dir.y << ", " << vertex.texCoord0Dir.z;
        vertices[i] = vertex;
    }

    return vertices;
}
