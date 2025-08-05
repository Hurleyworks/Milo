
#pragma once

static const char* MouseModeTable[] =
    {
        "Select",
        "Paint",
        "Rotate",
        "Translate",
        "Invalid"};

struct MouseMode
{
    enum EMouseMode
    {
        Select,
        Paint,
        Rotate,
        Translate,
        Count,
        Invalid = Count
    };

    union
    {
        EMouseMode name;
        unsigned int value;
    };

    MouseMode (EMouseMode name) :
        name (name) {}
    MouseMode (unsigned int value) :
        value (value) {}
    MouseMode() :
        value (Invalid) {}
    operator EMouseMode() const { return name; }
    const char* toString() const { return MouseModeTable[value]; }
    static MouseMode FromString (const char* str) { return mace::TableLookup (str, MouseModeTable, Count); }
};

class InputEvent
{
 public:
    enum struct Type
    {
        Idle = 0,
        Move,
        Press,
        Release,
        ScrollDown,
        ScrollUp,
        DblClick,
        KeyPress,
        KeyRelease,
        KeyRepeat,
        Drag
    };

    enum struct MouseButton
    {
        Left,
        Middle,
        Right,
        VScroll
    };

    enum struct Modifier
    {
        // changed to match GLFW
        None = 0,
        Shift = 1,
        Ctrl = 2,
        Alt = 4

    };

 public:
    InputEvent() = default;
    InputEvent (const Type& type,
                const MouseButton& button,
                float x,
                float y,
                bool alt = false,
                bool ctrl = false,
                bool shift = false) :
        type_ (type),
        button_ (button),
        mouseX_ (x),
        mouseY_ (y),
        keyState_ (0)
    {
        if (alt)
            keyState_ = static_cast<uint32_t> (Modifier::Alt);

        if (ctrl)
            keyState_ |= static_cast<uint32_t> (Modifier::Ctrl);

        if (shift)
            keyState_ |= static_cast<uint32_t> (Modifier::Shift);
    }
    ~InputEvent() = default;

    const MouseButton& getButton() const { return button_; }
    const MouseMode& getMouseMode() const { return mouseMode_; }
    void setMouseMode (MouseMode mode) { mouseMode_ = mode; }

    uint32_t getKeyboardModifiers() const { return keyState_; }
    const Type& getType() const { return type_; }
    float getX() const { return mouseX_; }
    float getY() const { return mouseY_; }
    void getScreenMovement (float& deltaX, float& deltaY)
    {
        deltaX = dx_;
        deltaY = dy_;
    }
    float getScreenMovementX() const { return dx_; }
    float getScreenMovementY() const { return dy_; }

    void setX (float x) { mouseX_ = x; }
    void setY (float y) { mouseY_ = y; }
    void setButton (MouseButton button) { button_ = button; }
    void setType (const Type& type) { type_ = type; }
    void setScreenMovement (float deltaX, float deltaY)
    {
        dx_ = deltaX;
        dy_ = deltaY;
    }
    void setKeyboardModifiers (unsigned int state) { keyState_ = state; }
    void setKey (uint32_t key) { key_ = key; }
    uint32_t getKey() const { return key_; }
    void setScanCode (uint32_t code) { scanCode = code; }
    uint32_t getScanCode() const { return scanCode; }

    void debug() const
    {
        switch (type_)
        {
            case InputEvent::Type::Release:
            {
                LOG (DBUG) << "Release";
                break;
            }
            case InputEvent::Type::Press:
            {
                LOG (DBUG) << "Press";
                break;
            }

            case InputEvent::Type::Move:
            {
                LOG (DBUG) << "Move";
                break;
            }

            case InputEvent::Type::Drag:
            {
                LOG (DBUG) << "Drag";
                break;
            }

            case InputEvent::Type::Idle:
            {
                LOG (DBUG) << "Idle";
                break;
            }

            case InputEvent::Type::KeyPress:
            {
                LOG (DBUG) << "Key Press";
                break;
            }

            case InputEvent::Type::KeyRelease:
            {
                LOG (DBUG) << "Key Release";
                break;
            }

            case InputEvent::Type::DblClick:
            {
                LOG (DBUG) << "Double click";
                break;
            }

            default:
                LOG (DBUG) << "FIXME";
                break;
        }
    }

 private:
    Type type_ = Type::Idle;
    Type prevType_ = Type::Idle;
    MouseButton button_ = MouseButton::Left;
    MouseMode mouseMode_ = MouseMode::Rotate;
    float mouseX_ = 0.0f;
    float mouseY_ = 0.0f;
    uint32_t keyState_ = 0;
    float dx_ = 0.0f;
    float dy_ = 0.0f;

    uint32_t key_ = ~0;
    uint32_t scanCode = ~0;

}; // end class InputEvent