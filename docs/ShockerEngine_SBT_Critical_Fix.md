# Critical Fix: ShockerEngine Hit Group SBT Buffer Initialization

## Problem Diagnosed
The CUDA debugging showed that:
- ✅ Ray Generation shader (`RT_RG_NAME(setupGBuffers)`) was being called
- ✅ Miss shader (`RT_MS_NAME(setupGBuffers)`) was being called  
- ❌ **Closest Hit shader (`RT_CH_NAME(setupGBuffers)`) was NEVER being called**

The rendered image showed a black cube silhouette against the gradient background, proving:
- Geometry WAS in the scene
- Rays WERE intersecting the geometry (blocking background)
- But the closest hit shader was NOT executing

## Root Cause
**INCORRECT ARGUMENT ORDER in Hit Group SBT buffer initialization!**

The code was using:
```cpp
// WRONG - this creates a buffer with 1 byte and hitGroupSbtSize elements!
hitGroupSbt.initialize(cuContext, cudau::BufferType::Device, 1, hitGroupSbtSize);
hitGroupSbt.resize(1, hitGroupSbtSize);
```

But the correct API signature is:
```cpp
// CORRECT - sizeInBytes comes BEFORE numElements
hitGroupSbt.initialize(cuContext, type, sizeInBytes, numElements);
hitGroupSbt.resize(sizeInBytes, numElements);
```

## The Fix
Changed all hit group SBT buffer operations to use the correct argument order:

### In `createSBT()`:
```cpp
// BEFORE (WRONG):
gbufferPipeline_->hitGroupSbt.initialize(cuContext, cudau::BufferType::Device, 1, hitGroupSbtSize);

// AFTER (CORRECT):
gbufferPipeline_->hitGroupSbt.initialize(cuContext, cudau::BufferType::Device, hitGroupSbtSize, 1);
```

### In `updateSBT()`:
```cpp
// BEFORE (WRONG):
gbufferPipeline_->hitGroupSbt.resize(1, hitGroupSbtSize);

// AFTER (CORRECT):
gbufferPipeline_->hitGroupSbt.resize(hitGroupSbtSize, 1);
```

## Why This Caused the Problem

1. The hit group SBT was being created with a size of **1 byte** instead of the required size (e.g., 256 bytes)
2. OptiX couldn't find the hit group records in the incorrectly sized buffer
3. When rays hit geometry, OptiX had no valid hit group to call
4. The result: rays would intersect geometry but not execute any shading

## Working Sample Confirmation
The working sample (`path_tracing_main.cpp`) uses the correct order:
```cpp
// Line 1712-1713 - CORRECT ORDER
gpuEnv.gBuffer.hitGroupSbt.initialize(
    gpuEnv.cuContext, Scene::bufferType, scene.hitGroupSbtSize, 1);
```

## Impact
This single parameter order mistake completely prevented all closest hit shaders from executing, making geometry appear black even though it was properly in the scene and intersecting rays.

## Files Modified
- `/mnt/e/1Milo/Milo/framework/engine_core/excludeFromBuild/engines/shocker/ShockerEngine.cpp`
  - Fixed in `createSBT()` function (4 locations)
  - Fixed in `updateSBT()` function (4 locations)

## Testing
After this fix:
- Build succeeded
- Hit group SBT now has correct size
- Closest hit shaders should now execute when rays intersect geometry
- Objects should render with proper shading

## Lesson Learned
Always verify buffer initialization argument order - especially when dealing with OptiX SBT buffers where the size is critical for proper shader dispatch!