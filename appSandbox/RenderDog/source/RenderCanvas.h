/*
    nanogui/imageview.h -- Widget used to display images.

    NanoGUI was developed by Wenzel Jakob <wenzel.jakob@epfl.ch>.
    The widget drawing code is based on the NanoVG demo application
    by Mikko Mononen.

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE.txt file.
*/
/** \file */

#pragma once

#include <sabi_core/sabi_core.h>

#include <date.h>
#include <nanovg.h>
#include <nanogui/screen.h>
#include <nanogui/layout.h>
#include <nanogui/window.h>
#include <nanogui/button.h>
#include <nanogui/canvas.h>
#include <nanogui/Label.h>
#include <nanogui/icons.h>
#include <nanogui/TextBox.h>
#include <nanogui/shader.h>
#include <nanogui/popupbutton.h>
#include <nanogui/imagepanel.h>
#include <nanogui/textbox.h>

#include <GLFW/glfw3.h>

using nanogui::Button;
using nanogui::FloatBox;
using nanogui::ImagePanel;
using nanogui::IntBox;
using nanogui::Label;
using nanogui::ref;
using nanogui::Shader;
using nanogui::TextBox;

using nanogui::Canvas;
using nanogui::Color;
using nanogui::Shader;
using nanogui::Texture;

using mace::InputEvent;
using mace::MouseMode;

// signal emitters
using InputSignal = Nano::Signal<void (const mace::InputEvent& e)>;

class RenderCanvas : public Canvas
{
 public:
    // signals
    InputSignal inputEmitter;

 public:
    RenderCanvas (Widget* parent, bool postProcess = true);

    void updateRender (const OIIO::ImageBuf& render, bool needsNewTexture = false)
    {
        if (imageTexture && !needsNewTexture)
        {
            imageTexture->upload ((uint8_t*)render.localpixels());
            set_image (imageTexture);
        }
        else if (needsNewTexture)
        {
            const OIIO::ImageSpec& spec = render.spec();
            size.x() = spec.width;
            size.y() = spec.height;
            OIIO::TypeDesc type = spec.format;

            LOG (DBUG) << "New screen size: " << spec.width << " x " << spec.height;

            imageTexture = new Texture (
                // Texture::PixelFormat::RGBA,
                // Texture::ComponentFormat::Float32,
                spec.nchannels == 3 ? Texture::PixelFormat::RGB : Texture::PixelFormat::RGBA,
                type == OIIO::TypeDesc::UINT8 ? Texture::ComponentFormat::UInt8 : Texture::ComponentFormat::Float32,
                size,
                Texture::InterpolationMode::Nearest,
                Texture::InterpolationMode::Nearest);

            imageTexture->upload ((uint8_t*)render.localpixels());
            set_image (imageTexture);
        }
    }

    /// Set the currently active image
    void set_image (Texture* image);

    bool mouse_button_event (const nanogui::Vector2i& p, int button, bool down, int modifiers) override;
    bool keyboard_event (int key, int scancode, int action, int modifiers) override;
    bool mouse_drag_event (const nanogui::Vector2i& p, const nanogui::Vector2i& rel, int button, int modifiers) override;
    bool scroll_event (const nanogui::Vector2i& p, const nanogui::Vector2f& rel) override;

    void draw (NVGcontext* ctx) override;
    void draw_contents() override;

    void setMouseMode (MouseMode mode) { mouseMode = mode; }
    MouseMode getMouseMode() const { return mouseMode; }
    void grabFrame()
    {
        captureFrame = true;
    }

 protected:
    nanogui::ref<Shader> imageShader;
    nanogui::ref<Texture> imageTexture;

    MouseMode mouseMode = MouseMode::Rotate;
    nanogui::Vector2i mousePressCoords = nanogui::Vector2i (-1, -1);

    float imageScale = 0;
    nanogui::Vector2f offset = 0;
    nanogui::Vector2i size;

    Color backgroundColor;
    bool captureFrame = false;

    InputEvent::MouseButton buttonPressed = InputEvent::MouseButton::Left;

    void captureAndSaveFrame()
    {
        // make sure to exclude the header and footer widgets from the screen grab
        uint32_t excludedGUIpixels = 0;
        // DEFAULT_GUI_FOOTER_HEIGHT + DEFAULT_GUI_HEADER_HEIGHT;

        OIIO::ImageSpec spec;
        spec.width = size.x();
        spec.height = size.y() - excludedGUIpixels;
        spec.nchannels = 4;
        spec.format = OIIO::TypeDesc::UINT8;

        // initialize an image bufferr to black pixels
        OIIO::ImageBuf render;
        render.reset (spec, OIIO::InitializePixels::Yes);

        // grab the rendered frame from GPU excluding the head and footer widget pixels
        glReadPixels (0, DEFAULT_GUI_HEADER_HEIGHT, size.x(), size.y() - excludedGUIpixels, GL_RGBA, GL_UNSIGNED_BYTE, render.localpixels());

        // make filename unique with current date, minute and second
        std::string date = date::format ("%F"
                                         "%M"
                                         "%S",
                                         std::chrono::system_clock::now());
        std::string filename = "/frame_grab_" + date + ".jpg";
        std::string frameGrabFolder = "";
        // properties.renderProps->getVal<std::string> (RenderKey::FramegrabFolder);
        std::string path = frameGrabFolder + filename;

        // must be flipped vertically
        OIIO::ImageBuf flipped = OIIO::ImageBufAlgo::flip (render);
        if (flipped.write (path.c_str(), OIIO::TypeDesc::UINT8))
        {
            LOG (DBUG) << "Saved screeen grab to " << path;
        }
        else
        {
            LOG (DBUG) << "Saving screen grab failed";
        }
    }
};
