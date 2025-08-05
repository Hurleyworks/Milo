

#include "berserkpch.h"
#include <GLFW/glfw3.h>
#include "OpenglRenderer.h"
#include <cassert>

// NanoGUI maintains a global map of all active screens indexed by their
// GLFW window handles. This map is defined in nanogui::screen.cpp and
// is populated whenever a new screen is created.
namespace nanogui
{
    extern std::map<GLFWwindow*, Screen*> __nanogui_screens;
}

// Constructor
OpenglRenderer::OpenglRenderer()
{
    // Verify that exactly one screen exists
    // This class is designed for single-screen applications only
    assert (nanogui::__nanogui_screens.size() == 1 &&
            "OpenglRenderer requires exactly one NanoGUI screen to exist");

    // Extract the window and screen from the global screens map
    // Since we've verified there's exactly one entry, we can safely
    // take the first (and only) entry
    for (const auto& it : nanogui::__nanogui_screens)
    {
        window = it.first;  // GLFW window handle
        screen = it.second; // NanoGUI screen instance
        break;              // Only process the first entry
    }

    // Ensure we successfully obtained valid pointers
    assert (window != nullptr && "Failed to obtain GLFW window handle");
    assert (screen != nullptr && "Failed to obtain NanoGUI screen instance");
}

// Window State Management
bool OpenglRenderer::isOpen()
{
    // GLFW returns 1 when the window should close, 0 when it should stay open
    // We invert this logic to provide a more intuitive interface
    return glfwWindowShouldClose (window) != 1;
}

// Event Handling
void OpenglRenderer::wait()
{
    // Block the current thread until at least one event is available
    // This is efficient for event-driven applications as it doesn't
    // consume CPU cycles while waiting for user input
    glfwWaitEvents();
}

void OpenglRenderer::refresh()
{
    // Request a screen redraw to update any changes
    screen->redraw();

    // Post an empty event to wake up any threads waiting in glfwWaitEvents()
    // This ensures that the main loop continues processing after the redraw
    glfwPostEmptyEvent();
}