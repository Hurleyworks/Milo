# Fix Summary: ShockerEngine Closest Hit Shader Not Being Called

## Problem
The closest hit shader in ShockerEngine was never being invoked during path tracing, causing geometry to not render.

## Root Cause
The ShockerEngine was missing a critical step: **it never registered the path tracing hit groups with the default material**.

While the engine properly:
- Created the hit program groups for path tracing (`RT_CH_NAME("shading")` and `RT_AH_NAME("visibility")`)
- Set up the pipelines correctly
- Built acceleration structures properly
- Set the traversable handle correctly

It failed to associate these hit groups with the material's ray types.

## The Missing Code
In `ShockerEngine.cpp`, after creating the path tracing pipeline programs, the code needed to register the hit groups with the default material for the appropriate ray types:

```cpp
// CRITICAL: Set path tracing hit groups on the default material
defaultMaterial_.setHitGroup(shocker_shared::PathTracingRayType::Closest,
                            pathTracePipeline_->hitPrograms.at(RT_CH_NAME_STR("shading")));
defaultMaterial_.setHitGroup(shocker_shared::PathTracingRayType::Visibility,
                            pathTracePipeline_->hitPrograms.at(RT_AH_NAME_STR("visibility")));
```

## Why This Matters
In OptiX, when a ray intersects geometry:
1. The ray type determines which hit group should be executed
2. The material associated with the geometry provides the mapping from ray type to hit group
3. Without this mapping, OptiX doesn't know which shader to execute for the intersection

The ShockerEngine only set up this mapping for the G-buffer pipeline's ray types, but not for the path tracing pipeline's ray types. This meant that when path tracing rays hit geometry, OptiX had no hit group to execute, effectively making the geometry invisible to path tracing.

## Verification
The working sample code (`PipelineManager.cpp:206-209`) clearly shows this step:
```cpp
optixDefaultMaterial.setHitGroup(
    shared::PathTracingRayType::Closest, 
    pipeline.hitPrograms.at(RT_CH_NAME_STR("pathTraceBaseline")));
optixDefaultMaterial.setHitGroup(
    shared::PathTracingRayType::Visibility, 
    pipeline.hitPrograms.at(RT_AH_NAME_STR("visibility")));
```

## Fix Applied
Added the missing `setHitGroup` calls in `ShockerEngine::createPrograms()` after the path tracing pipeline programs are created (lines 834-837).

## Result
The build succeeded with this fix. The closest hit shader should now be properly invoked when rays intersect geometry during path tracing.