#include "Jahley.h"

const std::string APP_NAME = "Eigen2Shocker";

#ifdef CHECK
#undef CHECK
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest/doctest.h>

#include <json/json.hpp>
using nlohmann::json;

#include <engine_core/engine_core.h>

// Include basic_types.h directly for Shocker math types
#include "../../../framework/engine_core/excludeFromBuild/common/basic_types.h"


// Helper function to check if two floats are approximately equal
bool approxEqual(float a, float b, float epsilon = 1e-6f) {
    return std::abs(a - b) < epsilon;
}

// Helper function to convert Eigen::Vector2f to Shocker float2
float2 eigenToShocker(const Eigen::Vector2f& v) {
    return float2(v.x(), v.y());
}

// Helper function to convert Eigen::Vector3f to Shocker float3
float3 eigenToShocker(const Eigen::Vector3f& v) {
    return float3(v.x(), v.y(), v.z());
}

// Helper function to convert Eigen::Vector4f to Shocker float4
float4 eigenToShocker(const Eigen::Vector4f& v) {
    return float4(v.x(), v.y(), v.z(), v.w());
}

// Helper function to convert Eigen::Vector2f to Shocker Vector2D
Vector2D eigenToShockerVec2D(const Eigen::Vector2f& v) {
    return Vector2D(v.x(), v.y());
}

// Helper function to convert Eigen::Vector3f to Shocker Vector3D
Vector3D eigenToShockerVec3D(const Eigen::Vector3f& v) {
    return Vector3D(v.x(), v.y(), v.z());
}

// Helper function to convert Eigen::Vector4f to Shocker Vector4D
Vector4D eigenToShockerVec4D(const Eigen::Vector4f& v) {
    return Vector4D(v.x(), v.y(), v.z(), v.w());
}

// Helper function to convert Eigen::Vector2f to Shocker Point2D
Point2D eigenToShockerPoint2D(const Eigen::Vector2f& v) {
    return Point2D(v.x(), v.y());
}

// Helper function to convert Eigen::Vector3f to Shocker Point3D
Point3D eigenToShockerPoint3D(const Eigen::Vector3f& v) {
    return Point3D(v.x(), v.y(), v.z());
}

// Helper function to convert Eigen::Matrix2f to Shocker Matrix2x2
Matrix2x2 eigenToShocker(const Eigen::Matrix2f& m) {
    // Eigen stores in column-major order, Shocker expects columns
    return Matrix2x2(
        Vector2D(m(0,0), m(1,0)),  // First column
        Vector2D(m(0,1), m(1,1))   // Second column
    );
}

// Helper function to convert Eigen::Matrix3f to Shocker Matrix3x3
Matrix3x3 eigenToShocker(const Eigen::Matrix3f& m) {
    // Eigen stores in column-major order, Shocker expects columns
    return Matrix3x3(
        Vector3D(m(0,0), m(1,0), m(2,0)),  // First column
        Vector3D(m(0,1), m(1,1), m(2,1)),  // Second column
        Vector3D(m(0,2), m(1,2), m(2,2))   // Third column
    );
}

// Helper function to convert Eigen::Matrix4f to Shocker Matrix4x4
Matrix4x4 eigenToShocker(const Eigen::Matrix4f& m) {
    // Eigen stores in column-major order, Shocker expects columns
    return Matrix4x4(
        Vector4D(m(0,0), m(1,0), m(2,0), m(3,0)),  // First column
        Vector4D(m(0,1), m(1,1), m(2,1), m(3,1)),  // Second column
        Vector4D(m(0,2), m(1,2), m(2,2), m(3,2)),  // Third column
        Vector4D(m(0,3), m(1,3), m(2,3), m(3,3))   // Fourth column
    );
}

// Helper function to convert Eigen::Affine3f to Shocker Matrix4x4
Matrix4x4 eigenToShocker(const Eigen::Affine3f& affine) {
    // Get the 4x4 matrix from the affine transformation
    Eigen::Matrix4f m = affine.matrix();
    return eigenToShocker(m);
}

// Helper function to convert Eigen::Quaternionf to Shocker Quaternion
Quaternion eigenToShocker(const Eigen::Quaternionf& q) {
    // Note: Eigen uses (x, y, z, w) ordering internally
    // Shocker Quaternion also uses (x, y, z, w) ordering
    return Quaternion(q.x(), q.y(), q.z(), q.w());
}

// Helper function to convert Shocker Quaternion to rotation matrix for testing
Matrix3x3 quaternionToMatrix3x3(const Quaternion& q) {
    float xx = q.x * q.x;
    float xy = q.x * q.y;
    float xz = q.x * q.z;
    float xw = q.x * q.w;
    float yy = q.y * q.y;
    float yz = q.y * q.z;
    float yw = q.y * q.w;
    float zz = q.z * q.z;
    float zw = q.z * q.w;
    
    return Matrix3x3(
        Vector3D(1.0f - 2.0f * (yy + zz), 2.0f * (xy + zw), 2.0f * (xz - yw)),
        Vector3D(2.0f * (xy - zw), 1.0f - 2.0f * (xx + zz), 2.0f * (yz + xw)),
        Vector3D(2.0f * (xz + yw), 2.0f * (yz - xw), 1.0f - 2.0f * (xx + yy))
    );
}

// Test suite for Vector conversions
TEST_CASE("Eigen to Shocker Vector Conversions") {
    
    SUBCASE("Vector2f to float2") {
        Eigen::Vector2f eigenVec(1.5f, 2.5f);
        float2 shockerVec = eigenToShocker(eigenVec);
        
        CHECK(approxEqual(shockerVec.x, 1.5f));
        CHECK(approxEqual(shockerVec.y, 2.5f));
    }
    
    SUBCASE("Vector3f to float3") {
        Eigen::Vector3f eigenVec(1.5f, 2.5f, 3.5f);
        float3 shockerVec = eigenToShocker(eigenVec);
        
        CHECK(approxEqual(shockerVec.x, 1.5f));
        CHECK(approxEqual(shockerVec.y, 2.5f));
        CHECK(approxEqual(shockerVec.z, 3.5f));
    }
    
    SUBCASE("Vector4f to float4") {
        Eigen::Vector4f eigenVec(1.5f, 2.5f, 3.5f, 4.5f);
        float4 shockerVec = eigenToShocker(eigenVec);
        
        CHECK(approxEqual(shockerVec.x, 1.5f));
        CHECK(approxEqual(shockerVec.y, 2.5f));
        CHECK(approxEqual(shockerVec.z, 3.5f));
        CHECK(approxEqual(shockerVec.w, 4.5f));
    }
    
    SUBCASE("Vector2f to Vector2D") {
        Eigen::Vector2f eigenVec(-1.0f, 2.0f);
        Vector2D shockerVec = eigenToShockerVec2D(eigenVec);
        
        CHECK(approxEqual(shockerVec.x, -1.0f));
        CHECK(approxEqual(shockerVec.y, 2.0f));
    }
    
    SUBCASE("Vector3f to Vector3D") {
        Eigen::Vector3f eigenVec(-1.0f, 2.0f, -3.0f);
        Vector3D shockerVec = eigenToShockerVec3D(eigenVec);
        
        CHECK(approxEqual(shockerVec.x, -1.0f));
        CHECK(approxEqual(shockerVec.y, 2.0f));
        CHECK(approxEqual(shockerVec.z, -3.0f));
    }
    
    SUBCASE("Vector4f to Vector4D") {
        Eigen::Vector4f eigenVec(-1.0f, 2.0f, -3.0f, 4.0f);
        Vector4D shockerVec = eigenToShockerVec4D(eigenVec);
        
        CHECK(approxEqual(shockerVec.x, -1.0f));
        CHECK(approxEqual(shockerVec.y, 2.0f));
        CHECK(approxEqual(shockerVec.z, -3.0f));
        CHECK(approxEqual(shockerVec.w, 4.0f));
    }
}

TEST_CASE("Eigen to Shocker Point Conversions") {
    
    SUBCASE("Vector2f to Point2D") {
        Eigen::Vector2f eigenVec(10.5f, 20.5f);
        Point2D shockerPoint = eigenToShockerPoint2D(eigenVec);
        
        CHECK(approxEqual(shockerPoint.x, 10.5f));
        CHECK(approxEqual(shockerPoint.y, 20.5f));
    }
    
    SUBCASE("Vector3f to Point3D") {
        Eigen::Vector3f eigenVec(10.5f, 20.5f, 30.5f);
        Point3D shockerPoint = eigenToShockerPoint3D(eigenVec);
        
        CHECK(approxEqual(shockerPoint.x, 10.5f));
        CHECK(approxEqual(shockerPoint.y, 20.5f));
        CHECK(approxEqual(shockerPoint.z, 30.5f));
    }
}

TEST_CASE("Eigen to Shocker Matrix Conversions") {
    
    SUBCASE("Matrix2f to Matrix2x2") {
        Eigen::Matrix2f eigenMat;
        eigenMat << 1.0f, 2.0f,
                    3.0f, 4.0f;
        
        Matrix2x2 shockerMat = eigenToShocker(eigenMat);
        
        // Check column 0
        CHECK(approxEqual(shockerMat.c0.x, 1.0f));
        CHECK(approxEqual(shockerMat.c0.y, 3.0f));
        
        // Check column 1
        CHECK(approxEqual(shockerMat.c1.x, 2.0f));
        CHECK(approxEqual(shockerMat.c1.y, 4.0f));
        
        // Check element access
        CHECK(approxEqual(shockerMat.m00, 1.0f));
        CHECK(approxEqual(shockerMat.m10, 3.0f));
        CHECK(approxEqual(shockerMat.m01, 2.0f));
        CHECK(approxEqual(shockerMat.m11, 4.0f));
    }
    
    SUBCASE("Matrix3f to Matrix3x3") {
        Eigen::Matrix3f eigenMat;
        eigenMat << 1.0f, 2.0f, 3.0f,
                    4.0f, 5.0f, 6.0f,
                    7.0f, 8.0f, 9.0f;
        
        Matrix3x3 shockerMat = eigenToShocker(eigenMat);
        
        // Check column 0
        CHECK(approxEqual(shockerMat.c0.x, 1.0f));
        CHECK(approxEqual(shockerMat.c0.y, 4.0f));
        CHECK(approxEqual(shockerMat.c0.z, 7.0f));
        
        // Check column 1
        CHECK(approxEqual(shockerMat.c1.x, 2.0f));
        CHECK(approxEqual(shockerMat.c1.y, 5.0f));
        CHECK(approxEqual(shockerMat.c1.z, 8.0f));
        
        // Check column 2
        CHECK(approxEqual(shockerMat.c2.x, 3.0f));
        CHECK(approxEqual(shockerMat.c2.y, 6.0f));
        CHECK(approxEqual(shockerMat.c2.z, 9.0f));
        
        // Check element access
        CHECK(approxEqual(shockerMat.m00, 1.0f));
        CHECK(approxEqual(shockerMat.m10, 4.0f));
        CHECK(approxEqual(shockerMat.m20, 7.0f));
        CHECK(approxEqual(shockerMat.m11, 5.0f));
        CHECK(approxEqual(shockerMat.m22, 9.0f));
    }
    
    SUBCASE("Matrix4f to Matrix4x4") {
        Eigen::Matrix4f eigenMat;
        eigenMat << 1.0f,  2.0f,  3.0f,  4.0f,
                    5.0f,  6.0f,  7.0f,  8.0f,
                    9.0f,  10.0f, 11.0f, 12.0f,
                    13.0f, 14.0f, 15.0f, 16.0f;
        
        Matrix4x4 shockerMat = eigenToShocker(eigenMat);
        
        // Check column 0
        CHECK(approxEqual(shockerMat.c0.x, 1.0f));
        CHECK(approxEqual(shockerMat.c0.y, 5.0f));
        CHECK(approxEqual(shockerMat.c0.z, 9.0f));
        CHECK(approxEqual(shockerMat.c0.w, 13.0f));
        
        // Check column 1
        CHECK(approxEqual(shockerMat.c1.x, 2.0f));
        CHECK(approxEqual(shockerMat.c1.y, 6.0f));
        CHECK(approxEqual(shockerMat.c1.z, 10.0f));
        CHECK(approxEqual(shockerMat.c1.w, 14.0f));
        
        // Check column 2
        CHECK(approxEqual(shockerMat.c2.x, 3.0f));
        CHECK(approxEqual(shockerMat.c2.y, 7.0f));
        CHECK(approxEqual(shockerMat.c2.z, 11.0f));
        CHECK(approxEqual(shockerMat.c2.w, 15.0f));
        
        // Check column 3
        CHECK(approxEqual(shockerMat.c3.x, 4.0f));
        CHECK(approxEqual(shockerMat.c3.y, 8.0f));
        CHECK(approxEqual(shockerMat.c3.z, 12.0f));
        CHECK(approxEqual(shockerMat.c3.w, 16.0f));
        
        // Check element access
        CHECK(approxEqual(shockerMat.m00, 1.0f));
        CHECK(approxEqual(shockerMat.m11, 6.0f));
        CHECK(approxEqual(shockerMat.m22, 11.0f));
        CHECK(approxEqual(shockerMat.m33, 16.0f));
    }
}

TEST_CASE("Identity Matrix Conversions") {
    
    SUBCASE("Identity Matrix2f to Matrix2x2") {
        Eigen::Matrix2f eigenMat = Eigen::Matrix2f::Identity();
        Matrix2x2 shockerMat = eigenToShocker(eigenMat);
        
        CHECK(approxEqual(shockerMat.m00, 1.0f));
        CHECK(approxEqual(shockerMat.m11, 1.0f));
        CHECK(approxEqual(shockerMat.m01, 0.0f));
        CHECK(approxEqual(shockerMat.m10, 0.0f));
    }
    
    SUBCASE("Identity Matrix3f to Matrix3x3") {
        Eigen::Matrix3f eigenMat = Eigen::Matrix3f::Identity();
        Matrix3x3 shockerMat = eigenToShocker(eigenMat);
        
        CHECK(approxEqual(shockerMat.m00, 1.0f));
        CHECK(approxEqual(shockerMat.m11, 1.0f));
        CHECK(approxEqual(shockerMat.m22, 1.0f));
        
        CHECK(approxEqual(shockerMat.m01, 0.0f));
        CHECK(approxEqual(shockerMat.m02, 0.0f));
        CHECK(approxEqual(shockerMat.m10, 0.0f));
        CHECK(approxEqual(shockerMat.m12, 0.0f));
        CHECK(approxEqual(shockerMat.m20, 0.0f));
        CHECK(approxEqual(shockerMat.m21, 0.0f));
    }
    
    SUBCASE("Identity Matrix4f to Matrix4x4") {
        Eigen::Matrix4f eigenMat = Eigen::Matrix4f::Identity();
        Matrix4x4 shockerMat = eigenToShocker(eigenMat);
        
        CHECK(approxEqual(shockerMat.m00, 1.0f));
        CHECK(approxEqual(shockerMat.m11, 1.0f));
        CHECK(approxEqual(shockerMat.m22, 1.0f));
        CHECK(approxEqual(shockerMat.m33, 1.0f));
        
        // Check all off-diagonal elements are zero
        CHECK(approxEqual(shockerMat.m01, 0.0f));
        CHECK(approxEqual(shockerMat.m02, 0.0f));
        CHECK(approxEqual(shockerMat.m03, 0.0f));
        CHECK(approxEqual(shockerMat.m10, 0.0f));
        CHECK(approxEqual(shockerMat.m12, 0.0f));
        CHECK(approxEqual(shockerMat.m13, 0.0f));
        CHECK(approxEqual(shockerMat.m20, 0.0f));
        CHECK(approxEqual(shockerMat.m21, 0.0f));
        CHECK(approxEqual(shockerMat.m23, 0.0f));
        CHECK(approxEqual(shockerMat.m30, 0.0f));
        CHECK(approxEqual(shockerMat.m31, 0.0f));
        CHECK(approxEqual(shockerMat.m32, 0.0f));
    }
}

TEST_CASE("Special Value Conversions") {
    
    SUBCASE("Zero vectors") {
        Eigen::Vector2f zeroVec2 = Eigen::Vector2f::Zero();
        Eigen::Vector3f zeroVec3 = Eigen::Vector3f::Zero();
        Eigen::Vector4f zeroVec4 = Eigen::Vector4f::Zero();
        
        float2 shockerVec2 = eigenToShocker(zeroVec2);
        float3 shockerVec3 = eigenToShocker(zeroVec3);
        float4 shockerVec4 = eigenToShocker(zeroVec4);
        
        CHECK(approxEqual(shockerVec2.x, 0.0f));
        CHECK(approxEqual(shockerVec2.y, 0.0f));
        
        CHECK(approxEqual(shockerVec3.x, 0.0f));
        CHECK(approxEqual(shockerVec3.y, 0.0f));
        CHECK(approxEqual(shockerVec3.z, 0.0f));
        
        CHECK(approxEqual(shockerVec4.x, 0.0f));
        CHECK(approxEqual(shockerVec4.y, 0.0f));
        CHECK(approxEqual(shockerVec4.z, 0.0f));
        CHECK(approxEqual(shockerVec4.w, 0.0f));
    }
    
    SUBCASE("Ones vectors") {
        Eigen::Vector2f onesVec2 = Eigen::Vector2f::Ones();
        Eigen::Vector3f onesVec3 = Eigen::Vector3f::Ones();
        Eigen::Vector4f onesVec4 = Eigen::Vector4f::Ones();
        
        float2 shockerVec2 = eigenToShocker(onesVec2);
        float3 shockerVec3 = eigenToShocker(onesVec3);
        float4 shockerVec4 = eigenToShocker(onesVec4);
        
        CHECK(approxEqual(shockerVec2.x, 1.0f));
        CHECK(approxEqual(shockerVec2.y, 1.0f));
        
        CHECK(approxEqual(shockerVec3.x, 1.0f));
        CHECK(approxEqual(shockerVec3.y, 1.0f));
        CHECK(approxEqual(shockerVec3.z, 1.0f));
        
        CHECK(approxEqual(shockerVec4.x, 1.0f));
        CHECK(approxEqual(shockerVec4.y, 1.0f));
        CHECK(approxEqual(shockerVec4.z, 1.0f));
        CHECK(approxEqual(shockerVec4.w, 1.0f));
    }
    
    SUBCASE("Unit vectors") {
        Eigen::Vector3f unitX = Eigen::Vector3f::UnitX();
        Eigen::Vector3f unitY = Eigen::Vector3f::UnitY();
        Eigen::Vector3f unitZ = Eigen::Vector3f::UnitZ();
        
        float3 shockerUnitX = eigenToShocker(unitX);
        float3 shockerUnitY = eigenToShocker(unitY);
        float3 shockerUnitZ = eigenToShocker(unitZ);
        
        CHECK(approxEqual(shockerUnitX.x, 1.0f));
        CHECK(approxEqual(shockerUnitX.y, 0.0f));
        CHECK(approxEqual(shockerUnitX.z, 0.0f));
        
        CHECK(approxEqual(shockerUnitY.x, 0.0f));
        CHECK(approxEqual(shockerUnitY.y, 1.0f));
        CHECK(approxEqual(shockerUnitY.z, 0.0f));
        
        CHECK(approxEqual(shockerUnitZ.x, 0.0f));
        CHECK(approxEqual(shockerUnitZ.y, 0.0f));
        CHECK(approxEqual(shockerUnitZ.z, 1.0f));
    }
}

TEST_CASE("Transformation Matrix Conversions") {
    
    SUBCASE("Translation Matrix") {
        // Create a translation matrix in Eigen
        Eigen::Matrix4f eigenTranslation = Eigen::Matrix4f::Identity();
        eigenTranslation(0, 3) = 10.0f;  // Translation in X
        eigenTranslation(1, 3) = 20.0f;  // Translation in Y
        eigenTranslation(2, 3) = 30.0f;  // Translation in Z
        
        Matrix4x4 shockerTranslation = eigenToShocker(eigenTranslation);
        
        // Check translation components
        CHECK(approxEqual(shockerTranslation.m03, 10.0f));
        CHECK(approxEqual(shockerTranslation.m13, 20.0f));
        CHECK(approxEqual(shockerTranslation.m23, 30.0f));
        
        // Check identity parts
        CHECK(approxEqual(shockerTranslation.m00, 1.0f));
        CHECK(approxEqual(shockerTranslation.m11, 1.0f));
        CHECK(approxEqual(shockerTranslation.m22, 1.0f));
        CHECK(approxEqual(shockerTranslation.m33, 1.0f));
    }
    
    SUBCASE("Scaling Matrix") {
        // Create a scaling matrix in Eigen
        Eigen::Matrix4f eigenScale = Eigen::Matrix4f::Identity();
        eigenScale(0, 0) = 2.0f;  // Scale X
        eigenScale(1, 1) = 3.0f;  // Scale Y
        eigenScale(2, 2) = 4.0f;  // Scale Z
        
        Matrix4x4 shockerScale = eigenToShocker(eigenScale);
        
        // Check scaling components
        CHECK(approxEqual(shockerScale.m00, 2.0f));
        CHECK(approxEqual(shockerScale.m11, 3.0f));
        CHECK(approxEqual(shockerScale.m22, 4.0f));
        CHECK(approxEqual(shockerScale.m33, 1.0f));
        
        // Check off-diagonal elements are zero
        CHECK(approxEqual(shockerScale.m01, 0.0f));
        CHECK(approxEqual(shockerScale.m02, 0.0f));
        CHECK(approxEqual(shockerScale.m10, 0.0f));
        CHECK(approxEqual(shockerScale.m12, 0.0f));
    }
    
    SUBCASE("Rotation Matrix Z-axis") {
        // Create a 45-degree rotation around Z-axis
        float angle = M_PI / 4.0f;  // 45 degrees
        Eigen::Matrix3f eigenRot3 = Eigen::Matrix3f::Identity();
        eigenRot3(0, 0) = cos(angle);
        eigenRot3(0, 1) = -sin(angle);
        eigenRot3(1, 0) = sin(angle);
        eigenRot3(1, 1) = cos(angle);
        
        Matrix3x3 shockerRot3 = eigenToShocker(eigenRot3);
        
        // Check rotation components
        CHECK(approxEqual(shockerRot3.m00, cos(angle)));
        CHECK(approxEqual(shockerRot3.m01, -sin(angle)));
        CHECK(approxEqual(shockerRot3.m10, sin(angle)));
        CHECK(approxEqual(shockerRot3.m11, cos(angle)));
        CHECK(approxEqual(shockerRot3.m22, 1.0f));
    }
}

TEST_CASE("Negative and Large Value Conversions") {
    
    SUBCASE("Negative values in vectors") {
        Eigen::Vector3f eigenVec(-100.5f, -200.25f, -300.75f);
        float3 shockerVec = eigenToShocker(eigenVec);
        
        CHECK(approxEqual(shockerVec.x, -100.5f));
        CHECK(approxEqual(shockerVec.y, -200.25f));
        CHECK(approxEqual(shockerVec.z, -300.75f));
    }
    
    SUBCASE("Large values in matrices") {
        Eigen::Matrix2f eigenMat;
        eigenMat << 1e6f, -1e6f,
                    1e7f, -1e7f;
        
        Matrix2x2 shockerMat = eigenToShocker(eigenMat);
        
        CHECK(approxEqual(shockerMat.m00, 1e6f, 1.0f));
        CHECK(approxEqual(shockerMat.m01, -1e6f, 1.0f));
        CHECK(approxEqual(shockerMat.m10, 1e7f, 10.0f));
        CHECK(approxEqual(shockerMat.m11, -1e7f, 10.0f));
    }
    
    SUBCASE("Mixed positive and negative values") {
        Eigen::Vector4f eigenVec(1.5f, -2.5f, 3.5f, -4.5f);
        Vector4D shockerVec = eigenToShockerVec4D(eigenVec);
        
        CHECK(approxEqual(shockerVec.x, 1.5f));
        CHECK(approxEqual(shockerVec.y, -2.5f));
        CHECK(approxEqual(shockerVec.z, 3.5f));
        CHECK(approxEqual(shockerVec.w, -4.5f));
    }
}

TEST_CASE("Precision and Edge Cases") {
    
    SUBCASE("Very small values") {
        Eigen::Vector3f eigenVec(1e-7f, 1e-8f, 1e-9f);
        float3 shockerVec = eigenToShocker(eigenVec);
        
        CHECK(approxEqual(shockerVec.x, 1e-7f, 1e-10f));
        CHECK(approxEqual(shockerVec.y, 1e-8f, 1e-11f));
        CHECK(approxEqual(shockerVec.z, 1e-9f, 1e-12f));
    }
    
    SUBCASE("Normalized vectors") {
        Eigen::Vector3f eigenVec(1.0f, 2.0f, 3.0f);
        eigenVec.normalize();
        
        Vector3D shockerVec = eigenToShockerVec3D(eigenVec);
        float length = sqrt(shockerVec.x * shockerVec.x + 
                          shockerVec.y * shockerVec.y + 
                          shockerVec.z * shockerVec.z);
        
        CHECK(approxEqual(length, 1.0f, 1e-5f));
    }
    
    SUBCASE("Transpose consistency") {
        Eigen::Matrix3f eigenMat;
        eigenMat << 1.0f, 2.0f, 3.0f,
                    4.0f, 5.0f, 6.0f,
                    7.0f, 8.0f, 9.0f;
        
        Eigen::Matrix3f eigenMatT = eigenMat.transpose();
        
        Matrix3x3 shockerMat = eigenToShocker(eigenMat);
        Matrix3x3 shockerMatT = eigenToShocker(eigenMatT);
        
        // Check that transpose relationship holds
        CHECK(approxEqual(shockerMat.m00, shockerMatT.m00));
        CHECK(approxEqual(shockerMat.m01, shockerMatT.m10));
        CHECK(approxEqual(shockerMat.m10, shockerMatT.m01));
        CHECK(approxEqual(shockerMat.m11, shockerMatT.m11));
    }
}

TEST_CASE("Eigen::Affine3f to Shocker Matrix4x4 Conversions") {
    
    SUBCASE("Identity Affine transformation") {
        Eigen::Affine3f eigenAffine = Eigen::Affine3f::Identity();
        Matrix4x4 shockerMat = eigenToShocker(eigenAffine);
        
        // Check identity matrix
        CHECK(approxEqual(shockerMat.m00, 1.0f));
        CHECK(approxEqual(shockerMat.m11, 1.0f));
        CHECK(approxEqual(shockerMat.m22, 1.0f));
        CHECK(approxEqual(shockerMat.m33, 1.0f));
        
        // Check off-diagonal elements are zero
        CHECK(approxEqual(shockerMat.m01, 0.0f));
        CHECK(approxEqual(shockerMat.m02, 0.0f));
        CHECK(approxEqual(shockerMat.m03, 0.0f));
        CHECK(approxEqual(shockerMat.m10, 0.0f));
        CHECK(approxEqual(shockerMat.m12, 0.0f));
        CHECK(approxEqual(shockerMat.m13, 0.0f));
        CHECK(approxEqual(shockerMat.m20, 0.0f));
        CHECK(approxEqual(shockerMat.m21, 0.0f));
        CHECK(approxEqual(shockerMat.m23, 0.0f));
        CHECK(approxEqual(shockerMat.m30, 0.0f));
        CHECK(approxEqual(shockerMat.m31, 0.0f));
        CHECK(approxEqual(shockerMat.m32, 0.0f));
    }
    
    SUBCASE("Translation Affine transformation") {
        Eigen::Affine3f eigenAffine = Eigen::Affine3f::Identity();
        eigenAffine.translation() = Eigen::Vector3f(5.0f, 10.0f, 15.0f);
        
        Matrix4x4 shockerMat = eigenToShocker(eigenAffine);
        
        // Check translation components
        CHECK(approxEqual(shockerMat.m03, 5.0f));
        CHECK(approxEqual(shockerMat.m13, 10.0f));
        CHECK(approxEqual(shockerMat.m23, 15.0f));
        
        // Check rotation/scale is identity
        CHECK(approxEqual(shockerMat.m00, 1.0f));
        CHECK(approxEqual(shockerMat.m11, 1.0f));
        CHECK(approxEqual(shockerMat.m22, 1.0f));
        CHECK(approxEqual(shockerMat.m33, 1.0f));
    }
    
    SUBCASE("Rotation Affine transformation") {
        // Create a rotation around Z-axis of 90 degrees
        Eigen::Affine3f eigenAffine = Eigen::Affine3f::Identity();
        eigenAffine.rotate(Eigen::AngleAxisf(M_PI / 2.0f, Eigen::Vector3f::UnitZ()));
        
        Matrix4x4 shockerMat = eigenToShocker(eigenAffine);
        
        // Check rotation components (90 degree rotation around Z)
        CHECK(approxEqual(shockerMat.m00, 0.0f, 1e-5f));  // cos(90)
        CHECK(approxEqual(shockerMat.m01, -1.0f, 1e-5f)); // -sin(90)
        CHECK(approxEqual(shockerMat.m10, 1.0f, 1e-5f));  // sin(90)
        CHECK(approxEqual(shockerMat.m11, 0.0f, 1e-5f));  // cos(90)
        CHECK(approxEqual(shockerMat.m22, 1.0f));
        CHECK(approxEqual(shockerMat.m33, 1.0f));
        
        // Check no translation
        CHECK(approxEqual(shockerMat.m03, 0.0f));
        CHECK(approxEqual(shockerMat.m13, 0.0f));
        CHECK(approxEqual(shockerMat.m23, 0.0f));
    }
    
    SUBCASE("Scale Affine transformation") {
        Eigen::Affine3f eigenAffine = Eigen::Affine3f::Identity();
        eigenAffine.scale(Eigen::Vector3f(2.0f, 3.0f, 4.0f));
        
        Matrix4x4 shockerMat = eigenToShocker(eigenAffine);
        
        // Check scale components
        CHECK(approxEqual(shockerMat.m00, 2.0f));
        CHECK(approxEqual(shockerMat.m11, 3.0f));
        CHECK(approxEqual(shockerMat.m22, 4.0f));
        CHECK(approxEqual(shockerMat.m33, 1.0f));
        
        // Check off-diagonal elements are zero
        CHECK(approxEqual(shockerMat.m01, 0.0f));
        CHECK(approxEqual(shockerMat.m02, 0.0f));
        CHECK(approxEqual(shockerMat.m10, 0.0f));
        CHECK(approxEqual(shockerMat.m12, 0.0f));
        CHECK(approxEqual(shockerMat.m20, 0.0f));
        CHECK(approxEqual(shockerMat.m21, 0.0f));
    }
    
    SUBCASE("Combined transformation") {
        // Create a complex transformation: translate, rotate, then scale
        Eigen::Affine3f eigenAffine = Eigen::Affine3f::Identity();
        eigenAffine.translate(Eigen::Vector3f(10.0f, 20.0f, 30.0f));
        eigenAffine.rotate(Eigen::AngleAxisf(M_PI / 4.0f, Eigen::Vector3f::UnitY()));
        eigenAffine.scale(2.0f);
        
        Matrix4x4 shockerMat = eigenToShocker(eigenAffine);
        
        // Check that translation is present
        CHECK(shockerMat.m03 != 0.0f);
        CHECK(shockerMat.m13 != 0.0f);
        CHECK(shockerMat.m23 != 0.0f);
        
        // Check that it's not identity (at least one diagonal element should be different from 1)
        bool isNotIdentity = (shockerMat.m00 != 1.0f) || (shockerMat.m11 != 1.0f) || (shockerMat.m22 != 1.0f);
        CHECK(isNotIdentity == true);
        
        // Check bottom row is still [0, 0, 0, 1]
        CHECK(approxEqual(shockerMat.m30, 0.0f));
        CHECK(approxEqual(shockerMat.m31, 0.0f));
        CHECK(approxEqual(shockerMat.m32, 0.0f));
        CHECK(approxEqual(shockerMat.m33, 1.0f));
    }
}

TEST_CASE("Eigen Block Extraction to Row-Major Matrix") {
    
    SUBCASE("Test 3x4 block extraction from 4x4 matrix") {
        // Create a test matrix with distinct values to track element positions
        Eigen::Matrix4f testMat;
        testMat << 1.0f,  2.0f,  3.0f,  4.0f,
                   5.0f,  6.0f,  7.0f,  8.0f,
                   9.0f,  10.0f, 11.0f, 12.0f,
                   13.0f, 14.0f, 15.0f, 16.0f;
        
        // Method 1: Using Eigen block extraction to row-major matrix (like our code)
        using MatrixRowMajor34f = Eigen::Matrix<float, 3, 4, Eigen::RowMajor>;
        MatrixRowMajor34f rowMajor34 = testMat.block<3, 4>(0, 0);
        
        // Method 2: Manual row-by-row extraction (like working sample)
        float manualArray[12] = {
            testMat(0,0), testMat(0,1), testMat(0,2), testMat(0,3),  // row 0
            testMat(1,0), testMat(1,1), testMat(1,2), testMat(1,3),  // row 1
            testMat(2,0), testMat(2,1), testMat(2,2), testMat(2,3)   // row 2
        };
        
        // Get the raw data from Eigen's row-major matrix
        const float* eigenData = rowMajor34.data();
        
        // Compare the two methods
        LOG(INFO) << "=== Eigen Block to Row-Major Test ===";
        LOG(INFO) << "Original Matrix (column-major):";
        for (int i = 0; i < 4; ++i) {
            LOG(INFO) << "  [" << testMat(i,0) << ", " << testMat(i,1) << ", " 
                      << testMat(i,2) << ", " << testMat(i,3) << "]";
        }
        
        LOG(INFO) << "Eigen Row-Major Data (sequential memory):";
        for (int i = 0; i < 12; ++i) {
            LOG(INFO) << "  eigenData[" << i << "] = " << eigenData[i];
        }
        
        LOG(INFO) << "Manual Row-Major Array:";
        for (int i = 0; i < 12; ++i) {
            LOG(INFO) << "  manualArray[" << i << "] = " << manualArray[i];
        }
        
        // Check if they match
        bool arraysMatch = true;
        for (int i = 0; i < 12; ++i) {
            if (!approxEqual(eigenData[i], manualArray[i])) {
                LOG(WARNING) << "Mismatch at index " << i << ": eigen=" << eigenData[i] 
                           << " manual=" << manualArray[i];
                arraysMatch = false;
            }
            CHECK(approxEqual(eigenData[i], manualArray[i]));
        }
        
        if (arraysMatch) {
            LOG(INFO) << "Arrays match - Eigen block extraction to row-major works correctly";
        } else {
            LOG (WARNING) << "Arrays don 't match - there' s a conversion problem ";
        }
    }
    
    SUBCASE("Test with translation matrix") {
        // Create a translation matrix (identity with translation in last column)
        Eigen::Matrix4f transMat = Eigen::Matrix4f::Identity();
        transMat(0, 3) = 10.0f;  // x translation
        transMat(1, 3) = 20.0f;  // y translation
        transMat(2, 3) = 30.0f;  // z translation
        
        using MatrixRowMajor34f = Eigen::Matrix<float, 3, 4, Eigen::RowMajor>;
        MatrixRowMajor34f rowMajor34 = transMat.block<3, 4>(0, 0);
        
        const float* data = rowMajor34.data();
        
        LOG(INFO) << "=== Translation Matrix Test ===";
        LOG(INFO) << "Translation should be in indices 3, 7, 11 for row-major";
        LOG(INFO) << "data[3] = " << data[3] << " (should be 10)";
        LOG(INFO) << "data[7] = " << data[7] << " (should be 20)";
        LOG(INFO) << "data[11] = " << data[11] << " (should be 30)";
        
        // In row-major 3x4, translation is at indices 3, 7, 11
        CHECK(approxEqual(data[3], 10.0f));
        CHECK(approxEqual(data[7], 20.0f));
        CHECK(approxEqual(data[11], 30.0f));
    }
    
    SUBCASE("Test column-major to row-major conversion explicitly") {
        // Create a matrix where we can easily see if it's transposed
        Eigen::Matrix4f testMat;
        testMat << 1.0f, 0.0f, 0.0f, 10.0f,   // column 0: basis X + translation X
                   0.0f, 1.0f, 0.0f, 20.0f,   // column 1: basis Y + translation Y  
                   0.0f, 0.0f, 1.0f, 30.0f,   // column 2: basis Z + translation Z
                   0.0f, 0.0f, 0.0f, 1.0f;    // column 3: homogeneous
        
        using MatrixRowMajor34f = Eigen::Matrix<float, 3, 4, Eigen::RowMajor>;
        MatrixRowMajor34f rowMajor34 = testMat.block<3, 4>(0, 0);
        
        // Also test the transpose
        Eigen::Matrix<float, 4, 3> transposed = testMat.block<3, 4>(0, 0).transpose();
        
        LOG(INFO) << "=== Column-Major to Row-Major Detailed Test ===";
        LOG(INFO) << "Original matrix element (0,3) [translation X]: " << testMat(0, 3);
        LOG(INFO) << "Row-major element (0,3): " << rowMajor34(0, 3);
        LOG(INFO) << "These should be equal if no transposition happened";
        
        // The (0,3) element should still be 10.0 if no transpose
        CHECK(approxEqual(rowMajor34(0, 3), 10.0f));
        CHECK(approxEqual(rowMajor34(1, 3), 20.0f));
        CHECK(approxEqual(rowMajor34(2, 3), 30.0f));
    }
}

TEST_CASE("Eigen::Quaternionf to Shocker Quaternion Conversions") {
    
    SUBCASE("Identity quaternion") {
        Eigen::Quaternionf eigenQuat = Eigen::Quaternionf::Identity();
        Quaternion shockerQuat = eigenToShocker(eigenQuat);
        
        CHECK(approxEqual(shockerQuat.x, 0.0f));
        CHECK(approxEqual(shockerQuat.y, 0.0f));
        CHECK(approxEqual(shockerQuat.z, 0.0f));
        CHECK(approxEqual(shockerQuat.w, 1.0f));
    }
    
    SUBCASE("90-degree rotation around X-axis") {
        // Create a 90-degree rotation around X-axis
        Eigen::Quaternionf eigenQuat(Eigen::AngleAxisf(M_PI / 2.0f, Eigen::Vector3f::UnitX()));
        Quaternion shockerQuat = eigenToShocker(eigenQuat);
        
        // For 90-degree rotation around X: q = [sin(45°), 0, 0, cos(45°)]
        float expected = sqrt(2.0f) / 2.0f;
        CHECK(approxEqual(shockerQuat.x, expected, 1e-5f));
        CHECK(approxEqual(shockerQuat.y, 0.0f, 1e-5f));
        CHECK(approxEqual(shockerQuat.z, 0.0f, 1e-5f));
        CHECK(approxEqual(shockerQuat.w, expected, 1e-5f));
    }
    
    SUBCASE("180-degree rotation around Y-axis") {
        // Create a 180-degree rotation around Y-axis
        Eigen::Quaternionf eigenQuat(Eigen::AngleAxisf(M_PI, Eigen::Vector3f::UnitY()));
        Quaternion shockerQuat = eigenToShocker(eigenQuat);
        
        // For 180-degree rotation around Y: q = [0, 1, 0, 0]
        CHECK(approxEqual(shockerQuat.x, 0.0f, 1e-5f));
        CHECK(approxEqual(shockerQuat.y, 1.0f, 1e-5f));
        CHECK(approxEqual(shockerQuat.z, 0.0f, 1e-5f));
        CHECK(approxEqual(shockerQuat.w, 0.0f, 1e-5f));
    }
    
    SUBCASE("45-degree rotation around Z-axis") {
        // Create a 45-degree rotation around Z-axis
        Eigen::Quaternionf eigenQuat(Eigen::AngleAxisf(M_PI / 4.0f, Eigen::Vector3f::UnitZ()));
        Quaternion shockerQuat = eigenToShocker(eigenQuat);
        
        // For 45-degree rotation around Z: q = [0, 0, sin(22.5°), cos(22.5°)]
        float sinHalf = sin(M_PI / 8.0f);
        float cosHalf = cos(M_PI / 8.0f);
        CHECK(approxEqual(shockerQuat.x, 0.0f, 1e-5f));
        CHECK(approxEqual(shockerQuat.y, 0.0f, 1e-5f));
        CHECK(approxEqual(shockerQuat.z, sinHalf, 1e-5f));
        CHECK(approxEqual(shockerQuat.w, cosHalf, 1e-5f));
    }
    
    SUBCASE("Arbitrary axis rotation") {
        // Create a rotation around an arbitrary normalized axis
        Eigen::Vector3f axis(1.0f, 1.0f, 1.0f);
        axis.normalize();
        float angle = M_PI / 3.0f; // 60 degrees
        
        Eigen::Quaternionf eigenQuat(Eigen::AngleAxisf(angle, axis));
        Quaternion shockerQuat = eigenToShocker(eigenQuat);
        
        // Verify quaternion is normalized
        float length = sqrt(shockerQuat.x * shockerQuat.x + 
                          shockerQuat.y * shockerQuat.y + 
                          shockerQuat.z * shockerQuat.z + 
                          shockerQuat.w * shockerQuat.w);
        CHECK(approxEqual(length, 1.0f, 1e-5f));
        
        // Verify the quaternion components match expected values
        float halfAngle = angle / 2.0f;
        float sinHalf = sin(halfAngle);
        float cosHalf = cos(halfAngle);
        
        CHECK(approxEqual(shockerQuat.x, axis.x() * sinHalf, 1e-5f));
        CHECK(approxEqual(shockerQuat.y, axis.y() * sinHalf, 1e-5f));
        CHECK(approxEqual(shockerQuat.z, axis.z() * sinHalf, 1e-5f));
        CHECK(approxEqual(shockerQuat.w, cosHalf, 1e-5f));
    }
    
    SUBCASE("Quaternion from rotation matrix") {
        // Create a rotation matrix and convert to quaternion
        Eigen::Matrix3f rotMat;
        rotMat = Eigen::AngleAxisf(M_PI / 6.0f, Eigen::Vector3f::UnitX()) *
                 Eigen::AngleAxisf(M_PI / 4.0f, Eigen::Vector3f::UnitY()) *
                 Eigen::AngleAxisf(M_PI / 3.0f, Eigen::Vector3f::UnitZ());
        
        Eigen::Quaternionf eigenQuat(rotMat);
        Quaternion shockerQuat = eigenToShocker(eigenQuat);
        
        // Verify quaternion is normalized
        float length = sqrt(shockerQuat.x * shockerQuat.x + 
                          shockerQuat.y * shockerQuat.y + 
                          shockerQuat.z * shockerQuat.z + 
                          shockerQuat.w * shockerQuat.w);
        CHECK(approxEqual(length, 1.0f, 1e-5f));
        
        // Convert back to matrix and compare key elements
        Matrix3x3 shockerRotMat = quaternionToMatrix3x3(shockerQuat);
        Matrix3x3 eigenRotMatConverted = eigenToShocker(rotMat);
        
        // Check that the matrices are approximately equal
        CHECK(approxEqual(shockerRotMat.m00, eigenRotMatConverted.m00, 1e-4f));
        CHECK(approxEqual(shockerRotMat.m11, eigenRotMatConverted.m11, 1e-4f));
        CHECK(approxEqual(shockerRotMat.m22, eigenRotMatConverted.m22, 1e-4f));
    }
    
    SUBCASE("Inverse quaternion") {
        // Create a quaternion and its inverse
        Eigen::Quaternionf eigenQuat(Eigen::AngleAxisf(M_PI / 3.0f, Eigen::Vector3f(1, 2, 3).normalized()));
        Eigen::Quaternionf eigenQuatInv = eigenQuat.inverse();
        
        Quaternion shockerQuat = eigenToShocker(eigenQuat);
        Quaternion shockerQuatInv = eigenToShocker(eigenQuatInv);
        
        // For unit quaternions, inverse is conjugate: (x, y, z, w) -> (-x, -y, -z, w)
        CHECK(approxEqual(shockerQuatInv.x, -shockerQuat.x, 1e-5f));
        CHECK(approxEqual(shockerQuatInv.y, -shockerQuat.y, 1e-5f));
        CHECK(approxEqual(shockerQuatInv.z, -shockerQuat.z, 1e-5f));
        CHECK(approxEqual(shockerQuatInv.w, shockerQuat.w, 1e-5f));
    }
}

class Application : public Jahley::App
{
public:
    Application(DesktopWindowSettings settings = DesktopWindowSettings(), bool windowApp = false) :
        Jahley::App()
    {
        doctest::Context().run();
    }

private:
};

Jahley::App* Jahley::CreateApplication()
{
    return new Application();
}

