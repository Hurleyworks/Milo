/*
    nanogui/imageview.cpp -- Widget used to display images.

    NanoGUI was developed by Wenzel Jakob <wenzel.jakob@epfl.ch>.
    The widget drawing code is based on the NanoVG demo application
    by Mikko Mononen.

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE.txt file.
*/

#include <nanogui/ext/nanovg/src/nanovg.h>
#include "RenderCanvas.h"
#include <nanogui/renderpass.h>

using nanogui::RenderPass;
using nanogui::Shader;
using nanogui::VariableType;
using nanogui::Vector3f;

// won't compile without this in Release mode but okay in Debug mode WTF?
inline Color::operator const NVGcolor&() const
{
    return reinterpret_cast<const NVGcolor&> (*(this->v));
}

RenderCanvas::RenderCanvas (Widget* parent, bool postProcess) :
    Canvas (parent, 1, false, false, false)

{
    render_pass()->set_clear_color (0, Color (0.3f, 0.3f, 0.32f, 1.f));

    try
    {
        if (postProcess)
        {
            imageShader = new Shader (
                render_pass(),

                "RenderShader",

                // Vertex shader
                R"(#version 330

            uniform mat4 matrix_image;
            uniform mat4 matrix_background;
            in vec2 position;
            out vec2 position_background;
            out vec2 uv;

            void main()
            {
            vec4 p = vec4(position, 0.0, 1.0);
            gl_Position = matrix_image * p;
            position_background = (matrix_background * p).xy;
            uv = position;
            })",

                // fragment shader
                // most of this shader code comes from LightHouse2 https://github.com/jbikker/lighthouse2
                R"(#version 330
            
            in vec2 uv;
            in vec2 position_background;
            out vec4 frag_color;
            uniform sampler2D image;
            uniform vec4 background_color;

            float luminance( vec3 v ) { return dot( v, vec3( 0.2126f, 0.7152f, 0.0722f ) ); }
            vec3 change_luminance( vec3 c_in, float l_out ) { float l_in = luminance( c_in ); return c_in * (l_out / l_in); }

            vec3 reinhard_jodie( vec3 v )
            {
	             float l = luminance( v );
	             vec3 tv = v / (1.0f + v);
	              return mix( v / (1.0f + l), tv, tv );
            }

            vec3 tonemap( vec3 v )
            {
                 return reinhard_jodie( v );
            }

            vec3 chromatic( vec2 UV )
            {
	            // chromatic abberation inspired by knifa (lsKSDz), via https://www.shadertoy.com/view/XlSBRW
	            vec2 d = abs( (UV - 0.5f) * 2.0f);    
	            d.x = pow( d.x, 1.5f );
	            d.y *= 0.1f;
	            float dScale = 0.01;
	            vec4 r, g, b;
	            r = g = b = vec4( 0.0 );
	            for (int i = 0; i < 2; ++i )
	            {
		            float rnd = i * 0.03f;
		            r += texture( image, UV + d * rnd * dScale );
		            g += texture( image, UV );
		            b += texture( image, UV - d * rnd * dScale );
	            }
	            return 0.5f * vec3( r.r, g.g, b.b );
            }

            float vignette( vec2 UV )
            {
                float vignetting = 0.35f;
	            // based on Keijiro Takahashi's Kino Vignette: https://github.com/keijiro/KinoVignette, via https://www.shadertoy.com/view/4lSXDm
	            vec2 coord = (UV - 0.5f) * 2.0f;
                float rf = sqrt( dot( coord, coord ) ) * vignetting;
                float rf2_1 = rf * rf + 1.0f;
                return 1.0f / (rf2_1 * rf2_1);
            }

            vec3 adjust( vec3 v )
            {
                float contrast = .025;
                float brightness = 0;
	            // https://www.dfstudios.co.uk/articles/programming/image-programming-algorithms/image-processing-algorithms-part-5-contrast-adjustment
	            float contrastFactor = (259.0f * (contrast * 256.0f + 255.0f)) / (255.0f * (259.0f - 256.0f * contrast));
	            float r = max( 0.0f, (v.x - 0.5f) * contrastFactor + 0.5f + brightness );
	            float g = max( 0.0f, (v.y - 0.5f) * contrastFactor + 0.5f + brightness );
	            float b = max( 0.0f, (v.z - 0.5f) * contrastFactor + 0.5f + brightness );

	            return vec3( r, g, b );
            }

            vec3 gamma_correct( vec3 v )
            {
	            float r = 1.0f / 2.2; // fixme hardwirded 2.2 for gamma now
	            return vec3( pow( v.x, r ), pow( v.y, r ), pow( v.z, r ) );
            }

            void main()
            {
                vec2 frac = position_background - floor(position_background);
                float checkerboard = ((frac.x > .5) == (frac.y > .5)) ? 0.4 : 0.5;

                vec4 background = (1.0 - background_color.a) * vec4(vec3(checkerboard), 1.0) +
                                  background_color.a * vec4(background_color.rgb, 1.0);

                vec4 value = texture(image, uv);
        
                // comment out these 2 lines to stop post processing for standard images
                // instead of renders coming from OptiX
                vec3 postProcess = gamma_correct( tonemap(adjust (vignette( uv ) *chromatic(uv)) ));
                value.rgb = postProcess.rgb;

                frag_color = (1.0 - value.a) * background + value.a * vec4(value.rgb, 1.0);
            })",

                Shader::BlendMode::AlphaBlend);
        }
        else
        {
            imageShader = new Shader (
                render_pass(),

                "RenderShader",

                // Vertex shader
                R"(#version 330

            uniform mat4 matrix_image;
            uniform mat4 matrix_background;
            in vec2 position;
            out vec2 position_background;
            out vec2 uv;

            void main()
            {
            vec4 p = vec4(position, 0.0, 1.0);
            gl_Position = matrix_image * p;
            position_background = (matrix_background * p).xy;
            uv = position;
            })",

                // fragment shader
                // most of this shader code comes from LightHouse2 https://github.com/jbikker/lighthouse2
                R"(#version 330
            
            in vec2 uv;
            in vec2 position_background;
            out vec4 frag_color;
            uniform sampler2D image;
            uniform vec4 background_color;

            float luminance( vec3 v ) { return dot( v, vec3( 0.2126f, 0.7152f, 0.0722f ) ); }
            vec3 change_luminance( vec3 c_in, float l_out ) { float l_in = luminance( c_in ); return c_in * (l_out / l_in); }

           
            void main()
            {
                vec2 frac = position_background - floor(position_background);
                float checkerboard = ((frac.x > .5) == (frac.y > .5)) ? 0.4 : 0.5;

                vec4 background = (1.0 - background_color.a) * vec4(vec3(checkerboard), 1.0) +
                                  background_color.a * vec4(background_color.rgb, 1.0);

                vec4 value = texture(image, uv);
        
                // comment out these 2 lines to stop post processing for standard images
                // instead of renders coming from OptiX
                vec3 postProcess = gamma_correct( tonemap(adjust (vignette( uv ) *chromatic(uv)) ));
               // value.rgb = postProcess.rgb;

               // frag_color = (1.0 - value.a) * background + value.a * vec4(value.rgb, 1.0);
            })",

                Shader::BlendMode::AlphaBlend);
        }
    }
    catch (std::exception& e)
    {
        LOG (CRITICAL) << e.what();
        return;
    }

    const float positions[] = {
        0.f, 0.f, 1.f, 0.f, 0.f, 1.f,
        1.f, 0.f, 1.f, 1.f, 0.f, 1.f};

    imageShader->set_buffer ("position", VariableType::Float32, {6, 2},
                             positions);
    m_render_pass->set_cull_mode (RenderPass::CullMode::Disabled);

    backgroundColor = Color (0.f, 0.f, 0.f, 0.f);

    size.x() = static_cast<int> (DEFAULT_DESKTOP_WINDOW_WIDTH);
    size.y() = static_cast<int> (DEFAULT_DESKTOP_WINDOW_HEIGHT);

    imageTexture = new nanogui::Texture (
        nanogui::Texture::PixelFormat::RGB,
        nanogui::Texture::ComponentFormat::UInt8,
        size,
        nanogui::Texture::InterpolationMode::Nearest,
        nanogui::Texture::InterpolationMode::Nearest);
}

void RenderCanvas::set_image (nanogui::Texture* image)
{
    if (image->mag_interpolation_mode() != nanogui::Texture::InterpolationMode::Nearest)
        throw std::runtime_error (
            "RenderCanvas::set_image(): interpolation mode must be set to 'Nearest'!");
    imageShader->set_texture ("image", image);
    imageTexture = image;
}

void RenderCanvas::draw (NVGcontext* ctx)
{
    if (!m_enabled || !imageTexture)
        return;

    Canvas::draw (ctx);
}

void RenderCanvas::draw_contents()
{
    if (!imageTexture)
        return;

    nanogui::Vector2i viewport_size = render_pass()->viewport().second;

    // Just fill the entire viewport - no scaling calculations needed
    nanogui::Matrix4f matrix_background =
        nanogui::Matrix4f::scale (Vector3f (viewport_size.x() / 20.f, viewport_size.y() / 20.f, 1.f));

    nanogui::Matrix4f matrix_image =
        nanogui::Matrix4f::ortho (0.f, viewport_size.x(), viewport_size.y(), 0.f, -1.f, 1.f) *
        nanogui::Matrix4f::scale (nanogui::Vector3f (viewport_size.x(), viewport_size.y(), 1.f));

    imageShader->set_uniform ("matrix_image", nanogui::Matrix4f (matrix_image));
    imageShader->set_uniform ("matrix_background", nanogui::Matrix4f (matrix_background));
    imageShader->set_uniform ("background_color", backgroundColor);

    imageShader->begin();
    imageShader->draw_array (Shader::PrimitiveType::Triangle, 0, 6, false);
    imageShader->end();

    if (captureFrame)
    {
        captureAndSaveFrame();
        captureFrame = false;
    }
}

bool RenderCanvas::keyboard_event (int key, int scancode, int action, int modifiers)
{
    if (!m_enabled || !imageTexture)
        return false;

    if (action == GLFW_PRESS)
    {
        if (key == GLFW_KEY_R)
        {
            // reset();
            return true;
        }
    }

    return false;
}

bool RenderCanvas::mouse_button_event (const nanogui::Vector2i& p, int button, bool down, int modifiers)
{
    InputEvent e{};

    switch (button)
    {
        case MOUSE_BUTTON_LEFT:
            e.setButton (InputEvent::MouseButton::Left);
            buttonPressed = InputEvent::MouseButton::Left;
            break;
        case MOUSE_BUTTON_RIGHT:
            e.setButton (InputEvent::MouseButton::Right);
            buttonPressed = InputEvent::MouseButton::Right;
            break;
        case MOUSE_BUTTON_MIDDLE:
            e.setButton (InputEvent::MouseButton::Middle);
            buttonPressed = InputEvent::MouseButton::Middle;
            break;
        default:
            break;
    }

    mousePressCoords.x() = p.x();
    mousePressCoords.y() = p.y();

    // FIXME what about double clicks
    e.setType (down ? InputEvent::Type::Press : InputEvent::Type::Release);

    e.setX (p.x());
    e.setY (p.y());

    e.setMouseMode (mouseMode);
    inputEmitter.fire (e);

    return false;
}

bool RenderCanvas::mouse_drag_event (const nanogui::Vector2i& p, const nanogui::Vector2i& rel,
                                     int button, int modifiers)
{
    InputEvent e{};

    e.setType (InputEvent::Type::Drag);
    e.setButton (buttonPressed);
    e.setX (p.x());
    e.setY (p.y());
    e.setKeyboardModifiers (modifiers);

    int deltaX = p.x() - mousePressCoords.x();
    int deltaY = p.y() - mousePressCoords.y();
    // LOG (DBUG) << modifiers;

    e.setMouseMode (mouseMode);
    e.setScreenMovement (deltaX, deltaY);

    inputEmitter.fire (e);

    return false;
}

bool RenderCanvas::scroll_event (const nanogui::Vector2i& p, const nanogui::Vector2f& rel)
{
    InputEvent e{};

    e.setType (rel.y() > 0.0 ? InputEvent::Type::ScrollUp : InputEvent::Type::ScrollDown);

    e.setMouseMode (mouseMode);
    inputEmitter.fire (e);
    return false;
}
