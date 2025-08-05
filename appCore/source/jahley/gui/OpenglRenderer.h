
#pragma once

#include <nanogui/screen.h>
#include <memory>

// Forward declarations
struct GLFWwindow;

// Type alias for cleaner API usage
using OpenglWindowHandle = std::unique_ptr<class OpenglRenderer>;

// OpenglRenderer Class
//
// Provides a simplified interface for managing OpenGL windows through NanoGUI.
// This class automatically discovers and manages the active NanoGUI screen,
// providing convenient methods for common window operations like rendering,
// event handling, and state checking.
class OpenglRenderer
{
 public:
    // Static Factory Method

    // Creates a new OpenglRenderer instance wrapped in a unique_ptr
    // Returns: Smart pointer to a new OpenglRenderer instance
    // Throws: std::exception if no NanoGUI screens are available or if
    //         multiple screens exist (single-screen limitation)
    static OpenglWindowHandle create() { return std::make_unique<OpenglRenderer>(); }

 public:

    // Constructs the renderer and automatically detects the active NanoGUI screen
    // Requires exactly one NanoGUI screen to exist at construction time
    OpenglRenderer();

    // Default destructor - resources are managed by NanoGUI/GLFW
    ~OpenglRenderer() = default;

    // Returns the underlying NanoGUI screen object
    // Returns: Pointer to the managed nanogui::Screen instance
    // Note: Pointer remains valid for the lifetime of this renderer
    nanogui::Screen* getScreen() { return screen; }

    // Checks if the window should remain open
    // Returns: true if window should stay open, false if close was requested
    bool isOpen();

    // Renders all screen elements to the framebuffer
    // This calls the NanoGUI screen's draw_all() method to render all widgets
    void render() { screen->draw_all(); }

    // Waits for GLFW events (blocks until events are available)
    // Use this in main loops to efficiently handle user input
    void wait();

    // Refreshes the screen and posts an empty event to wake up event loops
    // Call this after making changes that require a screen update
    void refresh();

 private:

    nanogui::Screen* screen = nullptr; // Pointer to the managed NanoGUI screen
    GLFWwindow* window = nullptr;      // Pointer to the underlying GLFW window

}; // end class OpenglRenderer