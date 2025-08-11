
#include <sabi_core/sabi_core.h>
#include "View.h"
#include <nanogui/window.h>
#include <nanogui/layout.h>
#include <nanogui/button.h>
#include <nanogui/combobox.h>
#include <nanogui/label.h>
#include <nanogui/checkbox.h>
#include <nanogui/slider.h>
#include <nanogui/textbox.h>

View::View (const DesktopWindowSettings& settings) :
    nanogui::Screen (nanogui::Vector2i (settings.width, settings.height), settings.name, settings.resizable),
    settings (settings)
{
    using namespace nanogui;
    inc_ref(); // or else we will crash.

    // create the default camera
    camera = std::make_shared<CameraBody>();
    camera->setFocalLength (0.055f); //. 55 mm lens
    camera->lookAt (Eigen::Vector3f (2.0f, 1.0f, 3.0f), Eigen::Vector3f (0.0f, 0.0f, 0.0f), Eigen::Vector3f (0.0f, 1.0f, 0.0f));

    // Create the canvas directly on the screen (no intermediate window)
    m_canvas = new RenderCanvas (this); // 'this' is the Screen
    m_canvas->set_background_color ({100, 100, 100, 255});
    m_canvas->set_fixed_size (Vector2i (width(), height()));

    // Set position to fill the screen
    m_canvas->set_position (Vector2i (0, 0));

    // Note: Legacy pipeline control window is hidden as we're using the new engine system
    // Create legacy pipeline control window but keep it hidden
    Window* pipelineWindow = new Window (this, "Pipeline Control");
    pipelineWindow->set_position (Vector2i (15, 15));
    pipelineWindow->set_layout (new GroupLayout());
    pipelineWindow->set_visible(false); // Hide legacy pipeline controls

    // Keep the pipeline combo for potential future use but hidden
    new Label (pipelineWindow, "Render Pipeline:", "sans-bold");
    m_pipelineCombo = new ComboBox (pipelineWindow,
                                    {"Realtime", "Quality", "Sequential (RT+Q)", "Scene", "Pick"});
    m_pipelineCombo->set_selected_index (0); // Default to Realtime
    m_pipelineCombo->set_tooltip ("Select the rendering pipeline to use");
    m_pipelineCombo->set_callback ([this] (int index)
                                   {
        const std::vector<std::string> pipelineNames = {
            "realtime",
            "quality",
            "sequential",
            "scene",
            "pick"
        };
        const std::vector<std::string> pipelineDescriptions = {
            "Fast rendering for interactive preview (2 bounces)",
            "High quality rendering with GI (8 bounces)",
            "Sequential execution: Realtime → Quality",
            "Basic scene rendering pipeline",
            "Object picking/selection pipeline"
        };
        if (index >= 0 && index < pipelineNames.size()) {
            LOG(INFO) << "Switching to pipeline: " << pipelineNames[index];
            onPipelineChange.fire(pipelineNames[index]);
            
            // Update the info label
            auto children = m_pipelineCombo->parent()->children();
            for (size_t i = 0; i < children.size(); i++) {
                if (auto label = dynamic_cast<Label*>(children[i])) {
                    if (label->caption() != "Pipeline Info:" && 
                        label->caption() != "Render Pipeline:" &&
                        label->caption() != "") {
                        label->set_caption(pipelineDescriptions[index]);
                        break;
                    }
                }
            }
        } });

    // Pipeline info label
    new Label (pipelineWindow, "Pipeline Info:", "sans-bold");
    Label* infoLabel = new Label (pipelineWindow, "Fast rendering for interactive preview (2 bounces)");
    infoLabel->set_font_size (14);

    // Add separator
    new Label (pipelineWindow, "", "sans-bold");

    // Enable pipeline system checkbox (hidden - always enabled)
    CheckBox* enablePipelineCheckbox = new CheckBox (pipelineWindow, "Enable Pipeline System");
    enablePipelineCheckbox->set_checked (true); // Default to enabled (engine system is active)
    enablePipelineCheckbox->set_tooltip ("Enable the new encapsulated pipeline system for rendering");
    enablePipelineCheckbox->set_callback ([this] (bool enabled)
                                          {
        LOG(INFO) << "Pipeline system " << (enabled ? "enabled" : "disabled");
        // Fire QMS message to enable/disable pipeline system
        onEnablePipelineSystem.fire(enabled); });

    // Create rendering engine control window
    Window* engineWindow = new Window (this, "Rendering Engine");
    engineWindow->set_position (Vector2i (15, 15)); // Position at top since pipeline window is hidden
    engineWindow->set_layout (new GroupLayout());

    // Engine selection combo box
    new Label (engineWindow, "Render Engine:", "sans-bold");
    m_engineCombo = new ComboBox (engineWindow,
                                  {"Milo Engine", "Claudia Engine", "RiPR Engine"});
    m_engineCombo->set_selected_index (0); // Default to Milo Engine
    m_engineCombo->set_tooltip ("Select the rendering engine to use");
    m_engineCombo->set_callback ([this] (int index)
                                 {
        const std::vector<std::string> engineNames = {
            "milo",
            "claudia",
            "ripr"
        };
        const std::vector<std::string> engineDescriptions = {
            "High-performance path tracing engine based on RiPR architecture",
            "Advanced path tracing engine with adaptive sampling and improved convergence",
            "Dual-pipeline ray tracing engine with G-buffer and path tracing modes"
        };
        if (index >= 0 && index < engineNames.size()) {
            LOG(INFO) << "Switching to engine: " << engineNames[index];
            onEngineChange.fire(engineNames[index]);
            
            // Show/hide engine-specific controls
            if (m_riprWindow) {
                m_riprWindow->set_visible(engineNames[index] == "ripr");
            }
            
            // Update the info label
            auto children = m_engineCombo->parent()->children();
            for (size_t i = 0; i < children.size(); i++) {
                if (auto label = dynamic_cast<Label*>(children[i])) {
                    if (label->caption() != "Engine Info:" && 
                        label->caption() != "Render Engine:" &&
                        label->caption() != "") {
                        label->set_caption(engineDescriptions[index]);
                        break;
                    }
                }
            }
        } });

    // Engine info label
    new Label (engineWindow, "Engine Info:", "sans-bold");
    Label* engineInfoLabel = new Label (engineWindow, "High-performance path tracing engine based on RiPR architecture");
    engineInfoLabel->set_font_size (14);

    // Create RiPR Engine controls window (initially hidden, shown when RiPR is selected)
    Window* riprWindow = new Window (this, "RiPR Engine Controls");
    riprWindow->set_position (Vector2i (15, 190)); // Position below engine window (same as Shocker)
    riprWindow->set_layout (new GroupLayout());
    riprWindow->set_visible(false); // Hidden by default
    
    // Render mode selection for RiPR Engine
    new Label (riprWindow, "Render Mode:", "sans-bold");
    ComboBox* riprRenderModeCombo = new ComboBox (riprWindow,
                                              {"Path Tracing", "G-Buffer Preview", "Debug Normals", 
                                               "Debug Albedo", "Debug Depth", "Debug Motion"});
    riprRenderModeCombo->set_selected_index (0); // Default to Path Tracing
    riprRenderModeCombo->set_tooltip ("Select the render mode for RiPR Engine");
    riprRenderModeCombo->set_callback ([this] (int index)
                                   {
        // Fire event for render mode change
        onRiPRRenderModeChange.fire(index); });
    
    // Store reference to RiPR window for visibility control
    m_riprWindow = riprWindow;

    // Create environment controls window
    Window* envWindow = new Window (this, "Environment Controls");
    envWindow->set_position (Vector2i (15, 300)); // Position below Shocker controls
    envWindow->set_layout (new GroupLayout());

    // Environment intensity label and slider
    new Label (envWindow, "Intensity:", "sans-bold");
    Widget* intensityPanel = new Widget (envWindow);
    intensityPanel->set_layout (new BoxLayout (Orientation::Horizontal, Alignment::Middle, 0, 10));
    
    Slider* intensitySlider = new Slider (intensityPanel);
    intensitySlider->set_value (1.0f); // Default 1.0 coefficient
    intensitySlider->set_range ({0.0f, 2.0f}); // 0-2 range
    intensitySlider->set_fixed_width (120);
    
    TextBox* intensityText = new TextBox (intensityPanel);
    intensityText->set_fixed_size (Vector2i (60, 25));
    intensityText->set_value ("1.00");
    intensityText->set_units ("");
    intensityText->set_font_size (16);
    intensityText->set_alignment (TextBox::Alignment::Right);
    
    // Add reset button for intensity
    Button* intensityResetBtn = new Button (intensityPanel, "Reset");
    intensityResetBtn->set_fixed_size (Vector2i (50, 25));
    intensityResetBtn->set_font_size (14);
    intensityResetBtn->set_callback ([this, intensitySlider, intensityText]() {
        // Reset to default intensity (1.0)
        intensitySlider->set_value (1.0f);
        intensityText->set_value ("1.00");
        onEnvironmentIntensityChange.fire (1.0f);
    });
    
    // Update text when slider changes and fire signal for real-time updates
    intensitySlider->set_callback ([this, intensityText](float value) {
        char buffer[32];
        snprintf(buffer, sizeof(buffer), "%.2f", value);
        intensityText->set_value (buffer);
        
        // Fire signal for real-time updates
        onEnvironmentIntensityChange.fire (value);
    });
    
    // Update slider when text changes
    intensityText->set_callback ([this, intensitySlider](const std::string& value) {
        try {
            float coefficient = std::stof(value);
            // Clamp to valid range
            coefficient = std::max(0.0f, std::min(2.0f, coefficient));
            intensitySlider->set_value (coefficient);
            
            // Fire signal when text is changed
            onEnvironmentIntensityChange.fire (coefficient);
        } catch (...) {
            // Invalid input, ignore
        }
        return true;
    });

    // Environment rotation label and slider
    new Label (envWindow, "Rotation:", "sans-bold");
    Widget* rotationPanel = new Widget (envWindow);
    rotationPanel->set_layout (new BoxLayout (Orientation::Horizontal, Alignment::Middle, 0, 10));
    
    Slider* rotationSlider = new Slider (rotationPanel);
    rotationSlider->set_value (0.5f); // Default 0 degrees (centered at 0.5)
    rotationSlider->set_range ({0.0f, 1.0f}); // Will map to -180 to 180 degrees
    rotationSlider->set_fixed_width (120);
    
    TextBox* rotationText = new TextBox (rotationPanel);
    rotationText->set_fixed_size (Vector2i (60, 25));
    rotationText->set_value ("0");
    rotationText->set_units ("°");
    rotationText->set_font_size (16);
    rotationText->set_alignment (TextBox::Alignment::Right);
    
    // Add reset button for rotation
    Button* rotationResetBtn = new Button (rotationPanel, "Reset");
    rotationResetBtn->set_fixed_size (Vector2i (50, 25));
    rotationResetBtn->set_font_size (14);
    rotationResetBtn->set_callback ([this, rotationSlider, rotationText]() {
        // Reset to default rotation (0 degrees)
        rotationSlider->set_value (0.5f); // 0.5 maps to 0 degrees
        rotationText->set_value ("0");
        onEnvironmentRotationChange.fire (0.0f);
    });
    
    // Update text when slider changes and fire signal for real-time updates
    // Store last value to implement threshold
    static float lastRotationValue = 0.5f;
    
    rotationSlider->set_callback ([this, rotationText](float value) {
        float degrees = (value - 0.5f) * 360.0f; // Map 0-1 to -180 to 180
        
        // Update text to show current value (rounded for display)
        rotationText->set_value (std::to_string(static_cast<int>(std::round(degrees))));
        
        // Only fire update if change is significant (reduces choppiness)
        static float lastFiredDegrees = 0.0f;
        float degreeDiff = std::abs(degrees - lastFiredDegrees);
        
        // Fire update if difference is at least 0.5 degrees
        if (degreeDiff >= 0.5f) {
            lastFiredDegrees = degrees;
            onEnvironmentRotationChange.fire (degrees);
        }
    });
    
    // Add final callback to ensure exact position when user releases
    rotationSlider->set_final_callback ([this](float value) {
        float degrees = (value - 0.5f) * 360.0f;
        onEnvironmentRotationChange.fire (degrees);
    });
    
    // Update slider when text changes
    rotationText->set_callback ([this, rotationSlider](const std::string& value) {
        try {
            float degrees = std::stof(value);
            // Clamp to -180 to 180
            degrees = std::max(-180.0f, std::min(180.0f, degrees));
            rotationSlider->set_value ((degrees / 360.0f) + 0.5f);
            
            // Fire signal when text is changed
            onEnvironmentRotationChange.fire (degrees);
        } catch (...) {
            // Invalid input, ignore
        }
        return true;
    });
    
    // Add a separator
    new Label (envWindow, "");
    
    // HDR file section
    new Label (envWindow, "HDR Image:", "sans-bold");
    
    // Current HDR filename display
    Widget* hdrPanel = new Widget (envWindow);
    hdrPanel->set_layout (new BoxLayout (Orientation::Horizontal, Alignment::Middle, 0, 5));
    
    Label* hdrFileLabel = new Label (hdrPanel, "None loaded");
    hdrFileLabel->set_font_size (14);
    hdrFileLabel->set_fixed_width (180);
    
    // Store the label as a member so we can update it later
    m_hdrFileLabel = hdrFileLabel;
    
    // Add info label about drag-drop
    Label* dragDropInfo = new Label (envWindow, "Drag & drop HDR/EXR files to load");
    dragDropInfo->set_font_size (12);
    dragDropInfo->set_color (Color(150, 150, 150, 255));
    
    // Add a separator
    new Label (envWindow, "");
    
    // Add Reset All button
    Button* resetAllBtn = new Button (envWindow, "Reset All");
    resetAllBtn->set_callback ([this, intensitySlider, intensityText, rotationSlider, rotationText]() {
        // Reset intensity to default (1.0)
        intensitySlider->set_value (1.0f);
        intensityText->set_value ("1.00");
        onEnvironmentIntensityChange.fire (1.0f);
        
        // Reset rotation to default (0 degrees)
        rotationSlider->set_value (0.5f); // 0.5 maps to 0 degrees
        rotationText->set_value ("0");
        onEnvironmentRotationChange.fire (0.0f);
    });

    // Create area light controls window
    Window* areaLightWindow = new Window (this, "Area Light Controls");
    areaLightWindow->set_position (Vector2i (15, 500)); // Position below environment window
    areaLightWindow->set_layout (new GroupLayout());
    
    // Enable area lights checkbox
    CheckBox* enableAreaLights = new CheckBox (areaLightWindow, "Enable Area Lights");
    enableAreaLights->set_checked (true); // Default to enabled
    enableAreaLights->set_tooltip ("Enable/disable area light sampling");
    enableAreaLights->set_callback ([this] (bool enabled)
    {
        LOG(INFO) << "Area lights " << (enabled ? "enabled" : "disabled");
        onAreaLightEnable.fire(enabled);
    });
    
    // Area light intensity label and controls
    new Label (areaLightWindow, "Area Light Power:", "sans-bold");
    Widget* areaLightPanel = new Widget (areaLightWindow);
    areaLightPanel->set_layout (new BoxLayout (Orientation::Horizontal, Alignment::Middle, 0, 10));
    
    Slider* areaLightSlider = new Slider (areaLightPanel);
    areaLightSlider->set_value (0.5f); // Default 10.0 (maps to 0.5)
    areaLightSlider->set_range ({0.0f, 1.0f}); // Will map to 0.1 to 100.0
    areaLightSlider->set_fixed_width (120);
    
    TextBox* areaLightText = new TextBox (areaLightPanel);
    areaLightText->set_fixed_size (Vector2i (60, 25));
    areaLightText->set_value ("10.0");
    areaLightText->set_font_size (16);
    areaLightText->set_alignment (TextBox::Alignment::Right);
    
    // Add reset button for area light intensity
    Button* areaLightResetBtn = new Button (areaLightPanel, "Reset");
    areaLightResetBtn->set_fixed_size (Vector2i (50, 25));
    areaLightResetBtn->set_font_size (14);
    areaLightResetBtn->set_callback ([this, areaLightSlider, areaLightText]() {
        // Reset to default intensity (10.0)
        areaLightSlider->set_value (0.5f); // 0.5 maps to 10.0
        areaLightText->set_value ("10.0");
        onAreaLightIntensityChange.fire (10.0f);
    });
    
    // Update text when slider changes
    areaLightSlider->set_callback ([this, areaLightText](float value) {
        // Map 0-1 to 0.1-100.0 using exponential scale for better control
        float intensity = 0.1f * std::pow(1000.0f, value); // 0.1 to 100.0
        
        // Update text to show current value
        areaLightText->set_value (std::to_string(static_cast<int>(std::round(intensity * 10.0f)) / 10.0f));
        
        // Fire update with some throttling
        static float lastFiredIntensity = 10.0f;
        float diff = std::abs(intensity - lastFiredIntensity);
        
        // Fire update if difference is significant
        if (diff >= 0.1f || intensity <= 0.2f || intensity >= 99.8f) {
            lastFiredIntensity = intensity;
            onAreaLightIntensityChange.fire (intensity);
        }
    });
    
    // Add final callback to ensure exact value when user releases
    areaLightSlider->set_final_callback ([this](float value) {
        float intensity = 0.1f * std::pow(1000.0f, value);
        onAreaLightIntensityChange.fire (intensity);
    });
    
    // Update slider when text changes
    areaLightText->set_callback ([this, areaLightSlider](const std::string& value) {
        try {
            float intensity = std::stof(value);
            // Clamp to 0.1 to 100.0
            intensity = std::max(0.1f, std::min(100.0f, intensity));
            // Convert to slider value using inverse of exponential mapping
            float sliderValue = std::log(intensity * 10.0f) / std::log(1000.0f);
            areaLightSlider->set_value (sliderValue);
            
            // Fire signal when text is changed
            onAreaLightIntensityChange.fire (intensity);
        } catch (...) {
            // Invalid input, ignore
        }
        return true;
    });
    
    // Add info label
    Label* areaLightInfo = new Label (areaLightWindow, "Controls emissive material brightness");
    areaLightInfo->set_font_size (12);
    areaLightInfo->set_color(Color(150, 150, 150, 255));

    // Create animation controls window
    Window* animWindow = new Window (this, "Animation Controls");
    animWindow->set_position (Vector2i (15, 680)); // Position below area light window
    animWindow->set_layout (new GroupLayout());

    // Y-axis rotation animation checkbox
    CheckBox* rotateAnimCheckbox = new CheckBox (animWindow, "Rotate Objects (Y-axis)");
    rotateAnimCheckbox->set_checked (false); // Default to off
    rotateAnimCheckbox->set_tooltip ("Enable continuous rotation of objects around their Y-axis");
    rotateAnimCheckbox->set_callback ([this] (bool enabled)
    {
        LOG(INFO) << "Object rotation animation " << (enabled ? "enabled" : "disabled");
        onAnimationToggle.fire(enabled);
    });

    // Finalize the layout
    perform_layout();
}

RenderCanvas* View::getCanvas()
{
    return m_canvas;
}

void View::debug()
{
    LOG (DBUG) << "Requested logical size: " << settings.width << "x" << settings.height;
    LOG (DBUG) << "Actual screen size: " << size().x() << "x" << size().y();
    LOG (DBUG) << "Framebuffer size: " << framebuffer_size().x() << "x" << framebuffer_size().y();
    LOG (DBUG) << "Pixel ratio: " << pixel_ratio();

    LOG (DBUG) << "Screen size: " << width() << "x" << height();
    LOG (DBUG) << "Canvas size: " << m_canvas->width() << "x" << m_canvas->height();
    LOG (DBUG) << "Canvas position: " << m_canvas->position().x() << "," << m_canvas->position().y();
}

void View::setHDRFilename(const std::string& filename)
{
    if (m_hdrFileLabel)
    {
        // Extract just the filename from the path
        std::filesystem::path p(filename);
        std::string displayName = p.filename().string();
        
        // Truncate if too long
        if (displayName.length() > 25)
        {
            displayName = displayName.substr(0, 22) + "...";
        }
        
        m_hdrFileLabel->set_caption(displayName);
    }
}

bool View::keyboard_event (int key, int scancode, int action, int modifiers)
{
    // Let parent class handle the event first
    if (Screen::keyboard_event (key, scancode, action, modifiers))
        return true;

    // Handle escape key to close the application
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
    {
        set_visible (false);
        return true;
    }

    return false;
}

void View::draw (NVGcontext* ctx)
{
    // Draw the complete user interface including canvas and controls
    Screen::draw (ctx);
}