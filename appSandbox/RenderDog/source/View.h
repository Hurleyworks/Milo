#pragma once

#include "RenderCanvas.h"

using sabi::CameraBody;
using sabi::CameraHandle;

using DropSignal = Nano::Signal<void (const std::vector<std::string>&)>;
using PipelineChangeSignal = Nano::Signal<void (const std::string&)>;
using EnablePipelineSystemSignal = Nano::Signal<void (bool)>;
using EngineChangeSignal = Nano::Signal<void (const std::string&)>;
using EnvironmentIntensityChangeSignal = Nano::Signal<void (float)>;
using EnvironmentRotationChangeSignal = Nano::Signal<void (float)>;
using HDRFileChangeSignal = Nano::Signal<void (const std::filesystem::path&)>;
using AnimationToggleSignal = Nano::Signal<void (bool)>;
using AreaLightIntensityChangeSignal = Nano::Signal<void (float)>;
using AreaLightEnableSignal = Nano::Signal<void (bool)>;
using RiPRRenderModeChangeSignal = Nano::Signal<void (int)>;

class View : public nanogui::Screen
{
 public:
    DropSignal onDrop;
    PipelineChangeSignal onPipelineChange;
    EnablePipelineSystemSignal onEnablePipelineSystem;
    EngineChangeSignal onEngineChange;
    EnvironmentIntensityChangeSignal onEnvironmentIntensityChange;
    EnvironmentRotationChangeSignal onEnvironmentRotationChange;
    HDRFileChangeSignal onHDRFileChange;
    AnimationToggleSignal onAnimationToggle;
    AreaLightIntensityChangeSignal onAreaLightIntensityChange;
    AreaLightEnableSignal onAreaLightEnable;
    RiPRRenderModeChangeSignal onRiPRRenderModeChange;

 public:
    // Constructor creates the main window with canvas and control buttons
    View (const DesktopWindowSettings& settings);

    // Get access to the canvas for external manipulation
    RenderCanvas* getCanvas();

    CameraHandle getCamera() const { return camera; }

    void debug();

    float pixelRatio() { return pixel_ratio(); }

    // Handle keyboard input events
    virtual bool keyboard_event (int key, int scancode, int action, int modifiers) override;

    // Main drawing function for the entire user interface
    virtual void draw (NVGcontext* ctx) override;

    virtual bool drop_event (const std::vector<std::string>& filenames) override
    {
        onDrop.fire (filenames);
        return false;
    }
    
    // Update the HDR filename display
    void setHDRFilename(const std::string& filename);

 private:
    CameraHandle camera = nullptr;

    DesktopWindowSettings settings;
    RenderCanvas* m_canvas;                       // The 3D rendering canvas containing the cube and camera
    nanogui::ComboBox* m_pipelineCombo = nullptr; // Pipeline selection combo box
    nanogui::ComboBox* m_engineCombo = nullptr;   // Engine selection combo box
    nanogui::Label* m_hdrFileLabel = nullptr;     // Label showing current HDR filename
    nanogui::Window* m_riprWindow = nullptr;      // RiPR Engine controls window
};