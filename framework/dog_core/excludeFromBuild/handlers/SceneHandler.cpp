#include "SceneHandler.h"
#include "../RenderContext.h"
#include <g3log/g3log.hpp>

namespace dog
{

SceneHandler::SceneHandler(RenderContextPtr ctx)
    : ctx_(ctx)
{
}

SceneHandler::~SceneHandler()
{
    finalize();
}

bool SceneHandler::initialize()
{
    if (initialized_)
    {
        LOG(WARNING) << "SceneHandler already initialized";
        return true;
    }

    if (!ctx_ || !ctx_->getCudaContext())
    {
        LOG(WARNING) << "SceneHandler: Invalid render context";
        return false;
    }

    try
    {
        // Initialize slot finders for resource allocation tracking
        material_slot_finder_.initialize(maxNumMaterials);
        geom_inst_slot_finder_.initialize(maxNumGeometryInstances);
        inst_slot_finder_.initialize(maxNumInstances);
        
        LOG(DBUG) << "SceneHandler slot finders initialized:";
        LOG(DBUG) << "  Max materials: " << maxNumMaterials;
        LOG(DBUG) << "  Max geometry instances: " << maxNumGeometryInstances;
        LOG(DBUG) << "  Max instances: " << maxNumInstances;
        
        // Initialize data buffers on device
        CUcontext cuContext = ctx_->getCudaContext();
        material_data_buffer_.initialize(cuContext, cudau::BufferType::Device, maxNumMaterials);
        geom_inst_data_buffer_.initialize(cuContext, cudau::BufferType::Device, maxNumGeometryInstances);
        inst_data_buffer_[0].initialize(cuContext, cudau::BufferType::Device, maxNumInstances);
        inst_data_buffer_[1].initialize(cuContext, cudau::BufferType::Device, maxNumInstances);
        
        LOG(DBUG) << "SceneHandler data buffers initialized";
        LOG(DBUG) << "  Material buffer size: " << maxNumMaterials;
        LOG(DBUG) << "  GeomInst buffer size: " << maxNumGeometryInstances;
        LOG(DBUG) << "  Instance buffers size: " << maxNumInstances << " (double buffered)";
        
        // Create Instance Acceleration Structure
        optixu::Context optixContext = ctx_->getOptixContext();
        optixu::Scene optixScene = optixContext.createScene();
        ias_ = optixScene.createInstanceAccelerationStructure();
        
        // Configure IAS for fast build (since we're interactive)
        ias_.setConfiguration(
            optixu::ASTradeoff::PreferFastBuild,
            optixu::AllowUpdate::Yes,
            optixu::AllowCompaction::No);
        
        LOG(DBUG) << "SceneHandler IAS created and configured";
        
        // Initialize light distribution for importance sampling
        // Using probability buffer approach (not texture)
        light_inst_dist_.initialize(cuContext, cudau::BufferType::Device, nullptr, maxNumInstances);
        LOG(DBUG) << "Light distribution initialized for " << maxNumInstances << " instances";
        
        // In an empty scene, traversable handle can be 0
        // This is a valid state for OptiX
        traversable_handle_ = 0;
        has_geometry_ = false;

        initialized_ = true;
        LOG(INFO) << "SceneHandler initialized successfully";
        return true;
    }
    catch (const std::exception& ex)
    {
        LOG(WARNING) << "Failed to initialize scene handler: " << ex.what();
        finalize();
        return false;
    }
}

void SceneHandler::finalize()
{
    if (!initialized_)
    {
        return;
    }

    // Clean up IAS and associated memory
    if (as_scratch_mem_.isInitialized())
    {
        as_scratch_mem_.finalize();
    }
    if (ias_instance_buffer_.isInitialized())
    {
        ias_instance_buffer_.finalize();
    }
    if (ias_mem_.isInitialized())
    {
        ias_mem_.finalize();
    }
    
    // Destroy IAS
    if (ias_)
    {
        ias_.destroy();
    }
    
    // Clean up data buffers
    inst_data_buffer_[1].finalize();
    inst_data_buffer_[0].finalize();
    geom_inst_data_buffer_.finalize();
    material_data_buffer_.finalize();
    
    // Clean up light distribution
    light_inst_dist_.finalize();
    
    // Clean up slot finders
    inst_slot_finder_.finalize();
    geom_inst_slot_finder_.finalize();
    material_slot_finder_.finalize();

    traversable_handle_ = 0;
    has_geometry_ = false;
    initialized_ = false;

    LOG(DBUG) << "SceneHandler finalized";
}

bool SceneHandler::buildAccelerationStructures()
{
    if (!initialized_)
    {
        LOG(WARNING) << "SceneHandler not initialized";
        return false;
    }

    try
    {
        // Check if we have any instances
        uint32_t numInstances = ias_.getNumChildren();
        
        if (numInstances == 0)
        {
            // Empty scene is valid - traversable handle stays 0
            traversable_handle_ = 0;
            has_geometry_ = false;
            LOG(DBUG) << "Empty scene - traversable handle = 0";
            return true;
        }
        
        // Get build requirements
        OptixAccelBufferSizes bufferSizes;
        ias_.prepareForBuild(&bufferSizes);
        
        // Allocate or resize scratch memory
        if (as_scratch_mem_.isInitialized())
        {
            if (bufferSizes.tempSizeInBytes > as_scratch_mem_.sizeInBytes())
            {
                as_scratch_mem_.resize(bufferSizes.tempSizeInBytes, 1);
            }
        }
        else
        {
            as_scratch_mem_.initialize(ctx_->getCudaContext(), cudau::BufferType::Device,
                                      bufferSizes.tempSizeInBytes, 1);
        }
        
        // Allocate or resize IAS memory
        if (ias_mem_.isInitialized())
        {
            if (bufferSizes.outputSizeInBytes > ias_mem_.sizeInBytes())
            {
                ias_mem_.resize(bufferSizes.outputSizeInBytes, 1);
            }
        }
        else
        {
            ias_mem_.initialize(ctx_->getCudaContext(), cudau::BufferType::Device, 
                               bufferSizes.outputSizeInBytes, 1);
        }
        
        // Allocate or resize instance buffer
        if (ias_instance_buffer_.isInitialized())
        {
            if (numInstances > ias_instance_buffer_.numElements())
            {
                ias_instance_buffer_.resize(numInstances);
            }
        }
        else
        {
            ias_instance_buffer_.initialize(ctx_->getCudaContext(), cudau::BufferType::Device, numInstances);
        }
        
        // Build the IAS
        CUstream stream = ctx_->getCudaStream();
        ias_.rebuild(stream, ias_instance_buffer_, ias_mem_, as_scratch_mem_);
        
        // Get traversable handle
        traversable_handle_ = ias_.getHandle();
        has_geometry_ = true;
        ias_needs_rebuild_ = false;
        
        LOG(INFO) << "IAS built successfully with " << numInstances << " instances";
        LOG(DBUG) << "Traversable handle: " << traversable_handle_;
        
        return true;
    }
    catch (const std::exception& ex)
    {
        LOG(WARNING) << "Failed to build acceleration structures: " << ex.what();
        return false;
    }
}

void SceneHandler::update()
{
    if (!initialized_)
    {
        LOG(WARNING) << "SceneHandler not initialized";
        return;
    }

    // Future implementation will handle:
    // - Transform updates
    // - Animation updates
    // - Dynamic object updates
    // - Visibility updates
    
    LOG(DBUG) << "Scene updated";
}

} // namespace dog