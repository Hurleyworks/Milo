
#pragma once

#include <properties_core/properties_core.h>

using mace::InputEvent;
using mace::MouseMode;
using sabi::CameraHandle;

using CmdSignal = Nano::Signal<void (void)>;
using InstanceOpSignal = Nano::Signal<void (uint32_t)>;
using OnModelVisibility = Nano::Signal<void(uint32_t mask)>;

class Controller
{
 public:
    // Signals
    CmdSignal framegrabEmitter;
    CmdSignal selectAllEmitter;
    CmdSignal deselectAllEmitter;
    InstanceOpSignal stackEmitter;
    OnModelVisibility modelVisibilityEmitter;

 public:
    Controller() = default;
    ~Controller() = default;

    void initialize (const PropertyService& props, CameraHandle cam) { 
        properties = props; 
        camera = cam;
    }
    void onInputEvent (const InputEvent& input, CameraHandle camera);
    
    // Environment control handlers
    void onEnvironmentIntensityChange(float intensity);
    void onEnvironmentRotationChange(float rotation);
    
    // Animation control handlers
    void onAnimationToggle(bool enable) { animationEnabled = enable; }
    bool isAnimationEnabled() const { return animationEnabled; }
    
    // Area light control handlers
    void onAreaLightIntensityChange(float intensity);
    void onAreaLightEnable(bool enable);

 private:
    Eigen::Vector2f mouseCoords;
    Eigen::Vector2f startMouseCoords;
    InputEvent previousInput;
    InputEvent::MouseButton buttonPressed = InputEvent::MouseButton::Left;
    PropertyService properties;
    CameraHandle camera = nullptr;

    float startScreenX = 0.0f;
    float startScreenY = 0.0f;
    bool animationEnabled = false;
}; // end class Controller
