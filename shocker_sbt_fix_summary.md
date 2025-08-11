# ShockerEngine SBT Fix Summary

## Problem
The closest hit shaders (both G-buffer and path tracing) were not being called because the Shader Binding Table (SBT) was empty or incorrectly sized.

## Root Cause
The scene's SBT layout was being generated BEFORE instances were added to the scene. This meant:
1. `scene_.generateShaderBindingTableLayout()` was returning a size of 0 or minimal size
2. The hit group SBTs were empty, so OptiX had no hit programs to call when rays intersected geometry
3. Even though the traversable handle was correct and rays were being traced, there were no shaders bound to execute

## Fixes Applied

### 1. Fixed Path Tracing Hit Group Registration (Previous Fix)
In `ShockerEngine::createPrograms()`, added registration of path tracing hit groups:
```cpp
defaultMaterial_.setHitGroup(shocker_shared::PathTracingRayType::Closest,
                            pathTracePipeline_->hitPrograms.at(RT_CH_NAME_STR("shading")));
defaultMaterial_.setHitGroup(shocker_shared::PathTracingRayType::Visibility,
                            pathTracePipeline_->hitPrograms.at(RT_AH_NAME_STR("visibility")));
```

### 2. Fixed SBT Generation Timing
In `ShockerSceneHandler::buildAccelerationStructures()`:
- Moved SBT layout generation to AFTER instances are added to the scene
- Added at line 428-432:
```cpp
// CRITICAL: Generate SBT layout AFTER instances are added to the scene
// This ensures the scene knows about all geometry for proper SBT generation
size_t hitGroupSbtSize;
scene_->generateShaderBindingTableLayout(&hitGroupSbtSize);
LOG(INFO) << "Generated scene SBT layout after adding instances, size: " << hitGroupSbtSize << " bytes";
```

### 3. Added SBT Update After Scene Clear
In `ShockerEngine::clearScene()`:
- Added `updateSBT()` call after clearing the scene to ensure SBT is properly reset

## How It Works Now

1. **When geometry is added** (`addGeometry()`):
   - Scene handler processes the node and creates instances
   - Instances are added to the scene via `scene_->createInstance()`
   - After all instances are added, `scene_->generateShaderBindingTableLayout()` is called
   - This generates the correct SBT size based on the actual geometry
   - `updateSBT()` is called to resize and update the pipeline SBTs

2. **When scene is cleared** (`clearScene()`):
   - Scene is cleared through the scene handler
   - `updateSBT()` is called to reset the SBTs to empty state

3. **The SBT now contains**:
   - Proper hit group entries for each geometry instance
   - Correct mappings from ray types to hit programs
   - Valid size based on actual scene content

## Result
With these fixes, when rays intersect geometry:
1. OptiX can find the appropriate hit group in the SBT
2. The closest hit shader is properly invoked
3. Both G-buffer and path tracing pipelines can execute their shaders

## Key Lesson
In an interactive application where geometry is added dynamically, the SBT must be regenerated/updated after the scene structure changes, not just during initialization. The scene needs to know about all instances before it can generate a correct SBT layout.