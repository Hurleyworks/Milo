#pragma once

// LightDistribution.h
// Provides importance sampling distribution for area lights
// Can use either probability texture (GPU-friendly) or buffer-based approach

#include "common_host.h"

namespace engine
{

// Simple light distribution using buffers
// In the future, we can add ProbabilityTexture support for better GPU performance
class LightDistribution
{
public:
    LightDistribution() : m_isInitialized(false), m_integral(0.0f), m_numValues(0) {}
    
    ~LightDistribution()
    {
        finalize();
    }
    
    // Move constructor
    LightDistribution(LightDistribution&& r) noexcept
    {
        m_weightBuffer = std::move(r.m_weightBuffer);
        m_cdfBuffer = std::move(r.m_cdfBuffer);
        m_integral = r.m_integral;
        m_numValues = r.m_numValues;
        m_isInitialized = r.m_isInitialized;
        r.m_isInitialized = false;
    }
    
    // Move assignment
    LightDistribution& operator=(LightDistribution&& r) noexcept
    {
        if (this != &r)
        {
            finalize();
            m_weightBuffer = std::move(r.m_weightBuffer);
            m_cdfBuffer = std::move(r.m_cdfBuffer);
            m_integral = r.m_integral;
            m_numValues = r.m_numValues;
            m_isInitialized = r.m_isInitialized;
            r.m_isInitialized = false;
        }
        return *this;
    }
    
    // Initialize with maximum number of values
    void initialize(CUcontext cuContext, uint32_t maxValues)
    {
        if (m_isInitialized)
        {
            return;
        }
        
        m_cuContext = cuContext;
        m_numValues = maxValues;
        
        // Allocate GPU buffers for weights and CDF
        m_weightBuffer.initialize(cuContext, cudau::BufferType::Device, maxValues);
        m_cdfBuffer.initialize(cuContext, cudau::BufferType::Device, maxValues);
        
        // Initialize to zero
        CUDADRV_CHECK(cuMemsetD32(reinterpret_cast<CUdeviceptr>(m_weightBuffer.getDevicePointer()), 0, maxValues));
        CUDADRV_CHECK(cuMemsetD32(reinterpret_cast<CUdeviceptr>(m_cdfBuffer.getDevicePointer()), 0, maxValues));
        
        m_isInitialized = true;
    }
    
    // Clean up resources
    void finalize()
    {
        if (!m_isInitialized)
        {
            return;
        }
        
        m_weightBuffer.finalize();
        m_cdfBuffer.finalize();
        m_integral = 0.0f;
        m_numValues = 0;
        m_isInitialized = false;
    }
    
    // Check if initialized
    bool isInitialized() const { return m_isInitialized; }
    
    // Set weight values
    void setWeights(const float* weights, uint32_t count, CUstream stream = 0)
    {
        if (!m_isInitialized || count > m_numValues)
        {
            return;
        }
        
        // Copy weights to GPU
        CUDADRV_CHECK(cuMemcpyHtoDAsync(
            reinterpret_cast<CUdeviceptr>(m_weightBuffer.getDevicePointer()),
            weights,
            sizeof(float) * count,
            stream));
        
        // Compute CDF on CPU for now (can be moved to GPU later)
        std::vector<float> cdf(count);
        float sum = 0.0f;
        for (uint32_t i = 0; i < count; ++i)
        {
            sum += weights[i];
            cdf[i] = sum;
        }
        
        // Normalize CDF
        if (sum > 0.0f)
        {
            float invSum = 1.0f / sum;
            for (uint32_t i = 0; i < count; ++i)
            {
                cdf[i] *= invSum;
            }
        }
        
        // Copy CDF to GPU
        CUDADRV_CHECK(cuMemcpyHtoDAsync(
            reinterpret_cast<CUdeviceptr>(m_cdfBuffer.getDevicePointer()),
            cdf.data(),
            sizeof(float) * count,
            stream));
        
        m_integral = sum;
    }
    
    // Set a single weight value
    void setWeightAt(uint32_t index, float weight, CUstream stream = 0)
    {
        if (!m_isInitialized || index >= m_numValues)
        {
            return;
        }
        
        CUDADRV_CHECK(cuMemcpyHtoDAsync(
            reinterpret_cast<CUdeviceptr>(m_weightBuffer.getDevicePointer()) + index * sizeof(float),
            &weight,
            sizeof(float),
            stream));
    }
    
    // Build probability texture (placeholder for future GPU optimization)
    void buildProbabilityTexture(CUstream stream = 0)
    {
        // TODO: Implement 2D probability texture for efficient GPU sampling
        // For now, we use the buffer-based approach
    }
    
    // Get the integral (sum of all weights)
    float integral() const { return m_integral; }
    
    // Get number of values
    uint32_t numValues() const { return m_numValues; }
    
    // Get weight buffer for GPU access
    const cudau::TypedBuffer<float>& getWeightBuffer() const { return m_weightBuffer; }
    cudau::TypedBuffer<float>& getWeightBuffer() { return m_weightBuffer; }
    
    // Get CDF buffer for GPU access
    const cudau::TypedBuffer<float>& getCDFBuffer() const { return m_cdfBuffer; }
    cudau::TypedBuffer<float>& getCDFBuffer() { return m_cdfBuffer; }
    
private:
    CUcontext m_cuContext = nullptr;
    cudau::TypedBuffer<float> m_weightBuffer;  // Individual weights
    cudau::TypedBuffer<float> m_cdfBuffer;     // Cumulative distribution function
    float m_integral;                          // Sum of all weights
    uint32_t m_numValues;                      // Number of values
    bool m_isInitialized;
};

} // namespace engine