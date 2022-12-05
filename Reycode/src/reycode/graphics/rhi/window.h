#pragma once

#include "reycode/reycode.h"

namespace reycode {
    struct RHI;
    struct Window;

    enum Key_State {
        KEY_UP,
        KEY_RELEASE,
        KEY_PRESS,
        KEY_REPEAT,
    };

    enum Mouse_Button {
        MOUSE_BUTTON_LEFT,
        MOUSE_BUTTON_RIGHT,
        MOUSE_BUTTON_MIDDLE,
        MOUSE_BUTTON_COUNT,
    };

    enum Key {
        KEY_LEFT_SHIFT,
        KEY_RIGHT_SHIFT,

        KEY_A,
        KEY_B,
        KEY_C,
        KEY_D,
        KEY_E,
        KEY_F,
        KEY_G,
        KEY_H,
        KEY_I,
        KEY_J,
        KEY_K,
        KEY_L,
        KEY_M,
        KEY_N,
        KEY_O,
        KEY_P,
        KEY_Q,
        KEY_R,
        KEY_S,
        KEY_T,
        KEY_U,
        KEY_V,
        KEY_W,
        KEY_X,
        KEY_Y,
        KEY_Z,

        KEY_ARROW_RIGHT,
        KEY_ARROW_LEFT,
        KEY_ARROW_DOWN,
        KEY_ARROW_UP,

        KEY_SPACE,

        KEY_COUNT
    };

    enum Key_Binding {
        KEY_BIND_UP,
        KEY_BIND_DOWN,
        KEY_BIND_LEFT,
        KEY_BIND_RIGHT,
        KEY_BIND_COUNT
    };

    struct Input_State {
        Key_State mouse_button_state[MOUSE_BUTTON_COUNT];
        Key_State key_state[KEY_COUNT];
        Key_State key_binding_state[KEY_BIND_COUNT];

        vec2 cursor_pos;
        vec2 cursor_delta;
    };

    struct Window_Input {
        struct Window* window;
        Input_State state;
        bool init;
    };

    bool input_mouse_button_down(const Input_State& state, Mouse_Button button);

    bool input_key_down(const Input_State& state, Key key);
    bool input_key_pressed(const Input_State& state, Key key);
    bool input_key_down(const Input_State& state, Key_Binding key);

    void window_input_poll(Window_Input& window);
    void window_input_capture_cursor(Window_Input& window, bool capture);

    struct Window_Desc {
        uint32_t width;
        uint32_t height;
        const char* title;
        bool vsync = false;
        bool validation = false;
    };

    Window* make_window(Arena& arena, const Window_Desc& desc);
    bool window_is_open(Window& window);
    void window_draw(Window& window);
    void destroy_window(Window* window);

    RHI& window_rhi(Window& window);

    double get_time();
}