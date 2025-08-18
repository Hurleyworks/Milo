#include "PipelineParameterHandler.h"
#include "../RenderContext.h"
#include "../GPUContext.h"

namespace dog
{

PipelineParameterHandler::PipelineParameterHandler(RenderContextPtr ctx)
    : render_context_(ctx)
{
}

PipelineParameterHandler::~PipelineParameterHandler()
{
    finalize();
}

bool PipelineParameterHandler::initialize()
{
    if (initialized_)
    {
        LOG(WARNING) << "PipelineParameterHandler already initialized";
        return true;
    }

    if (!render_context_)
    {
        LOG(WARNING) << "RenderContext is null";
        return false;
    }

    cuda_context_ = render_context_->getGPUContext().getCudaContext();

    try
    {
        // Initialize pick info buffers
        initializePickInfoBuffers();

        // Allocate device memory for pipeline parameters
        CUDADRV_CHECK(cuMemAlloc(&static_plp_on_device_, sizeof(static_plp_)));
        CUDADRV_CHECK(cuMemAlloc(&per_frame_plp_on_device_, sizeof(per_frame_plp_)));
        CUDADRV_CHECK(cuMemAlloc(&plp_on_device_, sizeof(plp_)));

        // Setup parameter pointers
        setupParameterPointers();

        // Initialize parameter structures to defaults
        static_plp_ = {};
        per_frame_plp_ = {};

        initialized_ = true;
        LOG(DBUG) << "PipelineParameterHandler initialized successfully";
        return true;
    }
    catch (const std::exception& ex)
    {
        LOG(WARNING) << "Failed to initialize pipeline parameters: " << ex.what();
        finalize();
        return false;
    }
}

void PipelineParameterHandler::finalize()
{
    if (!initialized_)
    {
        return;
    }

    // Free device memory
    if (static_plp_on_device_)
    {
        CUDADRV_CHECK(cuMemFree(static_plp_on_device_));
        static_plp_on_device_ = 0;
    }
    if (per_frame_plp_on_device_)
    {
        CUDADRV_CHECK(cuMemFree(per_frame_plp_on_device_));
        per_frame_plp_on_device_ = 0;
    }
    if (plp_on_device_)
    {
        CUDADRV_CHECK(cuMemFree(plp_on_device_));
        plp_on_device_ = 0;
    }

    // Finalize pick info buffers
    pick_infos_[1].finalize();
    pick_infos_[0].finalize();

    initialized_ = false;
    cuda_context_ = nullptr;

    LOG(DBUG) << "PipelineParameterHandler finalized";
}

void PipelineParameterHandler::updateStaticParameters(uint32_t width, uint32_t height)
{
    if (!initialized_)
    {
        LOG(WARNING) << "PipelineParameterHandler not initialized";
        return;
    }

    // Set basic parameters
    static_plp_.imageSize = int2(width, height);

    // Set pick info pointers
    static_plp_.pickInfos[0] = pick_infos_[0].getDevicePointer();
    static_plp_.pickInfos[1] = pick_infos_[1].getDevicePointer();

    // Note: Screen buffer surface objects and scene data will be set by external handlers
    // via setScreenBufferSurfaces() and setStaticSceneData()
}

void PipelineParameterHandler::copyStaticParametersToDevice()
{
    if (!initialized_)
    {
        LOG(WARNING) << "PipelineParameterHandler not initialized";
        return;
    }

    CUDADRV_CHECK(cuMemcpyHtoD(static_plp_on_device_, &static_plp_, sizeof(static_plp_)));
}

void PipelineParameterHandler::setStaticSceneData(const shared::ROBuffer<shared::DisneyData>& materialBuffer,
                                                  const shared::ROBuffer<shared::InstanceData>* instanceBuffers,
                                                  const shared::ROBuffer<shared::GeometryInstanceData>& geometryBuffer)
{
    if (!initialized_)
    {
        LOG(WARNING) << "PipelineParameterHandler not initialized";
        return;
    }

    static_plp_.materialDataBuffer = materialBuffer;
    if (instanceBuffers)
    {
        static_plp_.instanceDataBufferArray[0] = instanceBuffers[0];
        static_plp_.instanceDataBufferArray[1] = instanceBuffers[1];
    }
    static_plp_.geometryInstanceDataBuffer = geometryBuffer;
}

void PipelineParameterHandler::setEnvironmentLighting(CUtexObject envLightTexture,
                                                      const shared::RegularConstantContinuousDistribution2D& envLightImportanceMap)
{
    if (!initialized_)
    {
        LOG(WARNING) << "PipelineParameterHandler not initialized";
        return;
    }

    static_plp_.envLightTexture = envLightTexture;
    static_plp_.envLightImportanceMap = envLightImportanceMap;
}

void PipelineParameterHandler::setLightDistribution(const shared::LightDistribution& lightDist)
{
    if (!initialized_)
    {
        LOG(WARNING) << "PipelineParameterHandler not initialized";
        return;
    }

    static_plp_.lightInstDist = lightDist;
}

void PipelineParameterHandler::updatePerFrameParameters(const DogShared::PerFramePipelineLaunchParameters& params)
{
    if (!initialized_)
    {
        LOG(WARNING) << "PipelineParameterHandler not initialized";
        return;
    }

    per_frame_plp_ = params;
}

void PipelineParameterHandler::copyParametersToDevice(CUstream stream)
{
    if (!initialized_)
    {
        LOG(WARNING) << "PipelineParameterHandler not initialized";
        return;
    }

    // Copy per-frame parameters to device
    CUDADRV_CHECK(cuMemcpyHtoDAsync(per_frame_plp_on_device_, &per_frame_plp_, sizeof(per_frame_plp_), stream));

    // Copy combined pipeline parameters to device
    CUDADRV_CHECK(cuMemcpyHtoDAsync(plp_on_device_, &plp_, sizeof(plp_), stream));
}

void PipelineParameterHandler::copyToPureCUDADevice(CUstream stream, CUdeviceptr pureCudaDevicePtr)
{
    if (!initialized_)
    {
        LOG(WARNING) << "PipelineParameterHandler not initialized";
        return;
    }

    CUDADRV_CHECK(cuMemcpyHtoDAsync(pureCudaDevicePtr, &plp_, sizeof(plp_), stream));
}

const cudau::TypedBuffer<DogShared::PickInfo>& PipelineParameterHandler::getPickInfo(uint32_t index) const
{
    if (!isValidPickInfoIndex(index))
    {
        LOG(WARNING) << "Invalid pick info index: " << index;
        // Return first buffer as fallback
        return pick_infos_[0];
    }
    return pick_infos_[index];
}

cudau::TypedBuffer<DogShared::PickInfo>& PipelineParameterHandler::getPickInfo(uint32_t index)
{
    if (!isValidPickInfoIndex(index))
    {
        LOG(WARNING) << "Invalid pick info index: " << index;
        // Return first buffer as fallback
        return pick_infos_[0];
    }
    return pick_infos_[index];
}

DogShared::PickInfo* PipelineParameterHandler::getPickInfoPointer(uint32_t index) const
{
    if (!isValidPickInfoIndex(index))
    {
        LOG(WARNING) << "Invalid pick info index: " << index;
        return nullptr;
    }
    return pick_infos_[index].isInitialized() ? pick_infos_[index].getDevicePointer() : nullptr;
}

void PipelineParameterHandler::setScreenBufferSurfaces(const optixu::NativeBlockBuffer2D<DogShared::GBuffer0Elements>* gbuffer0,
                                                       const optixu::NativeBlockBuffer2D<DogShared::GBuffer1Elements>* gbuffer1,
                                                       const optixu::NativeBlockBuffer2D<float4>& beautyAccum,
                                                       const optixu::NativeBlockBuffer2D<float4>& albedoAccum,
                                                       const optixu::NativeBlockBuffer2D<float4>& normalAccum,
                                                       const optixu::NativeBlockBuffer2D<shared::PCG32RNG>& rngBuffer)
{
    if (!initialized_)
    {
        LOG(WARNING) << "PipelineParameterHandler not initialized";
        return;
    }

    // Set G-buffer references
    if (gbuffer0)
    {
        static_plp_.GBuffer0[0] = gbuffer0[0];
        static_plp_.GBuffer0[1] = gbuffer0[1];
    }
    if (gbuffer1)
    {
        static_plp_.GBuffer1[0] = gbuffer1[0];
        static_plp_.GBuffer1[1] = gbuffer1[1];
    }

    // Set accumulation buffer references
    static_plp_.beautyAccumBuffer = beautyAccum;
    static_plp_.albedoAccumBuffer = albedoAccum;
    static_plp_.normalAccumBuffer = normalAccum;

    // Set RNG buffer reference
    static_plp_.rngBuffer = rngBuffer;
}

void PipelineParameterHandler::initializePickInfoBuffers()
{
    // Initialize pick info buffers with default values
    DogShared::PickInfo initPickInfo = {};
    initPickInfo.hit = false;
    initPickInfo.instSlot = 0xFFFFFFFF;
    initPickInfo.geomInstSlot = 0xFFFFFFFF;
    initPickInfo.matSlot = 0xFFFFFFFF;
    initPickInfo.primIndex = 0xFFFFFFFF;
    initPickInfo.positionInWorld = Point3D(0.0f);
    initPickInfo.albedo = RGB(0.0f);
    initPickInfo.emittance = RGB(0.0f);
    initPickInfo.normalInWorld = Normal3D(0.0f);

    // Use default buffer type for compatibility
    pick_infos_[0].initialize(cuda_context_, cudau::BufferType::Device, 1, initPickInfo);
    pick_infos_[1].initialize(cuda_context_, cudau::BufferType::Device, 1, initPickInfo);
}

void PipelineParameterHandler::setupParameterPointers()
{
    // Set up the pipeline parameter pointers
    plp_.s = reinterpret_cast<DogShared::StaticPipelineLaunchParameters*>(static_plp_on_device_);
    plp_.f = reinterpret_cast<DogShared::PerFramePipelineLaunchParameters*>(per_frame_plp_on_device_);
}

} // namespace dog