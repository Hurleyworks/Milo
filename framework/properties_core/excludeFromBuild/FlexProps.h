
#pragma once

const float DEFAULT_PARTICLE_RADIUS = 0.25f;

static const char* FlexRenderAPITable[] =
    {
        "CUDA",
        "D3D",
        "Invalid"};

struct FlexRenderAPI
{
    enum EFlexRenderAPI
    {
        CUDA,
        D3D,
        Count,
        Invalid = Count
    };

    union
    {
        EFlexRenderAPI name;
        unsigned int value;
    };

    FlexRenderAPI (EFlexRenderAPI name) :
        name (name) {}
    FlexRenderAPI (unsigned int value) :
        value (value) {}
    FlexRenderAPI() :
        value (Invalid) {}
    operator EFlexRenderAPI() const { return name; }
    const char* ToString() const { return FlexRenderAPITable[value]; }
};

static const char* FlexToolTable[] =
    {
        "SquirtGun",
        "WaterPaint",
        "Invalid"};

struct FlexTool
{
    enum EFlexTool
    {
        SquirtGun,
        WaterPaint,
        Count,
        Invalid = Count
    };

    union
    {
        EFlexTool name;
        unsigned int value;
    };

    FlexTool (EFlexTool name) :
        name (name) {}
    FlexTool (unsigned int value) :
        value (value) {}
    FlexTool() :
        value (Invalid) {}
    operator EFlexTool() const { return name; }
    const char* ToString() const { return FlexToolTable[value]; }
};

static const char* RelaxModeTable[] =
    {
        "Local",
        "Global",
        "Invalid"};

struct RelaxMode
{
    enum ERelaxMode
    {
        Local,
        Global,
        Count,
        Invalid = Count
    };

    union
    {
        ERelaxMode name;
        unsigned int value;
    };

    RelaxMode (ERelaxMode name) :
        name (name) {}
    RelaxMode (unsigned int value) :
        value (value) {}
    RelaxMode() :
        value (Invalid) {}
    operator ERelaxMode() const { return name; }
    const char* ToString() const { return RelaxModeTable[value]; }
};

static const char* FlexRenderModeTable[] =
    {
        "Preview",
        "Bake",
        "Invalid"};

struct FlexRenderMode
{
    enum EFlexRenderMode
    {
        Preview,
        Bake,
        Count,
        Invalid = Count
    };

    union
    {
        EFlexRenderMode name;
        unsigned int value;
    };

    FlexRenderMode (EFlexRenderMode name) :
        name (name) {}
    FlexRenderMode (unsigned int value) :
        value (value) {}
    FlexRenderMode() :
        value (Invalid) {}
    operator EFlexRenderMode() const { return name; }
    const char* ToString() const { return FlexRenderModeTable[value]; }
};

static const char* FlexKeyTable[] =
    {
        "drawEllipsoids",
        "drawPoints",
        "drawDiffuse",
        "drawMesh",
        "timeDelta",
        "simStep",
        "realWorldTimeDelta",
        "substeps",
        "iterations",
        "shutdown",
        "particleSize",
        "collisionDistance",
        "particleMargin",
        "kinematicMargin",
        "renderMode",
        "viewport",
        "viewportDirty",
        "force",
        "maxSpeed",
        "maxAccel",
        "maxParticleCount",
        "dissipation",
        "damping",
        "adhesion",
        "particleCount",
        "dynamicFriction",
        "staticFriction",
        "restitution",
        "sleepThreshold",
        "shockPropagation",
        "relaxMode",
        "relaxFactor",

        // fluids
        "viscosity",
        "cohesion",
        "surfaceTension",
        "vorticity",
        "buoyancy",
        "freeSurfaceDrag",
        "solidPressure",
        "fluidRestDistance",

        // fluid mesh modifiers
        "gaussian",
        "gaussianWidth",
        "laplacian",
        "dilate",
        "erode",
        "resize",
        "dilateIterations",
        "erodeIterations",
        "offset",

        // emitters
        "emitterSpeedMultiplier",
        "emitterWidth",
        "emitterEnabled",
        "emitterDelay",
        "emitterParticleCount",
        "emitterType",

        // remesh
        "remeshTarget",
        "remeshType",

        // tool
        "flexTool",

        "Invalid"};

struct FlexKey
{
    enum EFlexKey
    {
        drawEllipsoids,
        drawPoints,
        drawDiffuse,
        drawMesh,
        timeDelta,
        simStep,
        realWorldTimeDelta,
        substeps,
        iterations,
        shutdown,
        particleSize,
        collisionDistance,
        particleMargin,
        kinematicMargin,
        renderMode,
        viewport,
        viewportDirty,
        force,
        maxSpeed,
        maxAccel,
        maxParticleCount,
        dissipation,
        damping,
        adhesion,
        particleCount,
        dynamicFriction,
        staticFriction,
        restitution,
        sleepThreshold,
        shockPropagation,
        relaxMode,
        relaxFactor,

        // fluids
        viscosity,
        cohesion,
        surfaceTension,
        vorticity,
        buoyancy,
        freeSurfaceDrag,
        solidPressure,
        fluidRestDistance,

        // fluid mesh modifiers
        gaussian,
        gaussianWidth,
        laplacian,
        dilate,
        erode,
        resize,
        dilateIterations,
        erodeIterations,
        offset,

        // emitters
        emitterSpeedMultiplier,
        emitterWidth,
        emitterEnabled,
        emitterDelay,
        emitterParticleCount,
        emitterType,

        // remesh
        remeshTarget,
        remeshType,

        //
        flexTool,

        Count,
        Invalid = Count
    };

    union
    {
        EFlexKey name;
        unsigned int value;
    };

    FlexKey (EFlexKey name) :
        name (name) {}
    FlexKey (unsigned int value) :
        value (value) {}
    FlexKey() :
        value (Invalid) {}
    operator EFlexKey() const { return name; }
    const char* ToString() const { return FlexKeyTable[value]; }
};

// remeshing with instant meshes
const double DEFAULT_FLEX_REMESH_VERTEX_COUNT = 0.0;

const FlexRenderMode DEFAULT_FLEX_FLEX_RENDER_MODE = FlexRenderMode::Preview;
const double DEFAULT_FLEX_TIME_DELTA = 1.0 / 60.0;     // the time delta used for simulation
const float DEFAULT_FLEX_REAL_WORLD_TIME_DELTA = 0.0f; // the real world time delta
const int DEFAULT_FLEX_FLEX_SUBSTEPS = 2;
const int DEFAULT_FLEX_FLEX_ITERATIONS = 3;
const Eigen::Vector2i DEFAULT_FLEX_VIEWPORT_SIZE = Eigen::Vector2i::Zero();
const RelaxMode DEFAULT_FLEX_RELAX_MODE = RelaxMode::Local;
const double DEFAULT_FLEX_RELAX_FACTOR = 1.25;
const FlexTool DEFAULT_FLEX_TOOL = FlexTool::SquirtGun;

// these are Goo defaults
const double DEFAULT_FLEX_VISCOSITY = 1.0;
const double DEFAULT_FLEX_ADHESION = 00;
const double ADHESION_ADJUSTER = .01; // Adhesion seems to be too powerful
const double DEFAULT_FLEX_COHESION = 0.3;
const double DEFAULT_FLEX_SURFACE_TENSION = 0.0;
const double DEFAULT_FLEX_VORTICITY = 0;
const double DEFAULT_FLEX_BUOYANCY = 1;
const double DEFAULT_FLEX_FREE_SURFACE_DRAG = 0;
const double DEFAULT_FLEX_SOLID_PRESSURE = 0;
const double DEFAULT_FLEX_FLUID_REST_DISTANCE = .65;

// openvdb filters to change the shape of the mesh
const int DEFAULT_FLEX_GAUSSIAN = 1;
const int DEFAULT_FLEX_LAPLACIAN = 0;
const int DEFAULT_FLEX_DILATE = 0;
const int DEFAULT_FLEX_ERODE = 1;
const int DEFAULT_FLEX_RESIZE = 0;
const int DEFAULT_FLEX_DILATE_ITERATIONS = 1;
const int DEFAULT_FLEX_ERODE_ITERATIONS = 2;
const int DEFAULT_FLEX_GAUSSIAN_WIDTH = 1;
const double DEFAULT_FLEX_FILTER_OFFSET = 0.05f;
const int DEFAULT_FLEX_MAX_PARTICLE_COUNT = 64000;
const int DEFAULT_FLEX_PARTICLE_COUNT = DEFAULT_FLEX_MAX_PARTICLE_COUNT;
const double DEFAULT_FLEX_PARTICLE_SIZE = 0.25f;
const double DEFAULT_FLEX_COLLISION_DISTANCE = DEFAULT_FLEX_PARTICLE_SIZE * 0.5;
const double DEFAULT_FLEX_PARTICLE_MARGIN = 0;
const double DEFAULT_FLEX_KINEMATIC_MARGIN = 0;
const double DEFAULT_FLEX_RESTITUTION = 0;
const double DEFAULT_FLEX_SHOCK_PROPAGATION = 0;
const double DEFAULT_FLEX_SLEEP_THRESHOLD = 0;
const double STATIC_FRICTION_DEFAULT = 0;
const double DEFAULT_FLEX_DYNAMIC_FRICTION = .08;

const double DEFAULT_FLEX_EMITTER_WIDTH = 24.0;
const double DEFAULT_FLEX_EMITTER_ENABLED = 1.0;
const double DEFAULT_FLEX_EMITTER_SPEED_MULTIPLIER = 1.0;
const int DEFAULT_FLEX_EMITTER_PARTICLE_COUNT = DEFAULT_FLEX_PARTICLE_COUNT;

const Eigen::Vector3d DEFAULT_FLEX_FORCE = Eigen::Vector3d (0.0, -9.8, 0.0);

// Preview drawing
const int DEFAULT_FLEX_DRAW_ELLIPSOIDS = 1;
const int DEFAULT_FLEX_DRAW_DIFFUSE = 0;
const int DEFAULT_FLEX_DRAW_POINTS = 0;
const int DEFAULT_FLEX_DRAW_MESH = 0;

const double DEFAULT_FLEX_MAX_SPEED = 100;
const double DEFAULT_FLEX_MAX_ACCEL = 100;
const double DEFAULT_FLEX_DAMPING = 0;
const double DEFAULT_FLEX_DISSIPATION = 0;

const size_t DEFAULT_FLEX_FLEX_SCREEN_WIDTH = 1024;
const size_t DEFAULT_FLEX_FLEX_SCREEN_HEIGHT = 1024;

typedef AnyValue<FlexKey> FlexProperties;
using FlexPropsRef = std::shared_ptr<FlexProperties>;