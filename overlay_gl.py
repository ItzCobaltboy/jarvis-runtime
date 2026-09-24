"""
GPU-rendered version of overlay.py — same visuals and interface, but the
vignette flood (angular rainbow, exponential falloff, pulsing, state colors)
runs as a single fragment shader instead of ~200 CPU-built QLinearGradients
redrawn at 30fps. Meant to fix the battery/CPU cost of the QPainter version
on a laptop running the NPU transcription pipeline alongside it.

Only Python-side state (a handful of floats) is pushed to the GPU each frame;
all the per-pixel color math happens in GLSL. The pill (state dot + text) is
still drawn with QPainter on a small overlay layer — it's tiny and not worth
shader complexity.

--- main.py bootstrap (do not modify main.py — this documents the intended wiring) ---
#
# import sys
# from PyQt6.QtWidgets import QApplication
# from overlay_gl import JarvisOverlayGL as JarvisOverlay
# import threading
#
# qt_app = QApplication(sys.argv)
# overlay = JarvisOverlay()
#
# def audio_pipeline():
#     overlay.show_state(...)
#     ...
#
# threading.Thread(target=audio_pipeline, daemon=True).start()
# sys.exit(qt_app.exec())
#
overlay.show_state(...) is thread-safe, same as overlay.py.
---------------------------------------------------------------------------
"""

import sys

from PyQt6.QtCore import (
    Qt, QRectF, QPointF, QTimer,
    QPropertyAnimation, QParallelAnimationGroup, QEasingCurve,
    pyqtProperty, QMetaObject, Q_ARG, pyqtSlot,
)
from PyQt6.QtGui import QPainter, QPainterPath, QColor, QFont, QGuiApplication, QSurfaceFormat
from PyQt6.QtWidgets import QApplication
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from OpenGL import GL as gl

PILL_WIDTH = 240
PILL_HEIGHT = 30
PILL_MARGIN_TOP = 16
PILL_MARGIN_RIGHT = 16

BLEED_TOP = 0.10
BLEED_LEFT = 0.09

CENTER_TINT_ALPHA = 0.00001

EDGE_ALPHA_LISTENING = 75 / 255
EDGE_ALPHA_PROCESSING_MIN = 140.0 / 255
EDGE_ALPHA_PROCESSING_MAX = 190.0 / 255
EDGE_ALPHA_SUCCESS = 40 / 255
EDGE_ALPHA_UNRECOGNISED = 40 / 255

EDGE_DECAY_RATE = 12.0  # exponential falloff rate k in exp(-k*t); higher = sharper cutoff

COLOR_AMBER = (245 / 255, 158 / 255, 11 / 255)
COLOR_GREEN = (34 / 255, 197 / 255, 94 / 255)
COLOR_RED = (239 / 255, 68 / 255, 68 / 255)

STATE_TEXT = {
    "listening": "Listening...",
    "processing": "Thinking...",
    "unrecognised": "Didn't catch that",
}

AUTO_DISMISS_MS = 600

STATE_IDS = {"idle": 0, "listening": 1, "processing": 2, "success": 3, "unrecognised": 4}

VERTEX_SHADER = """
#version 330 core
layout(location = 0) in vec2 position;
out vec2 fragCoord;
void main() {
    fragCoord = (position + 1.0) * 0.5;  // NDC [-1,1] -> uv [0,1]
    gl_Position = vec4(position, 0.0, 1.0);
}
"""

FRAGMENT_SHADER = """
#version 330 core
in vec2 fragCoord;
out vec4 outColor;

uniform vec2 u_resolution;
uniform float u_floodOpacity;
uniform int u_state;          // "to" state (matches STATE_IDS)
uniform int u_prevState;      // "from" state, for cross-fading between states
uniform float u_crossfade;    // 0 = fully u_prevState, 1 = fully u_state
uniform float u_hueOffsetDeg; // rotates the angular rainbow over time
uniform float u_pulseAlpha;   // 0..1, used only in processing state
uniform float u_bleedTop;
uniform float u_bleedLeft;
uniform float u_edgeAlphaListening;
uniform float u_edgeAlphaSuccess;
uniform float u_edgeAlphaUnrecognised;
uniform float u_centerTint;
uniform vec3 u_colorAmber;
uniform vec3 u_colorGreen;
uniform vec3 u_colorRed;

const float PI = 3.14159265359;
const float EDGE_DECAY_RATE = __EDGE_DECAY_RATE__;

vec3 hsv2rgb(float h, float s, float v) {
    vec3 k = vec3(1.0, 2.0 / 3.0, 1.0 / 3.0);
    vec3 p = abs(fract(vec3(h) + k) * 6.0 - 3.0);
    return v * mix(vec3(1.0), clamp(p - 1.0, 0.0, 1.0), s);
}

// Exponential falloff matching the CPU version's k=4 decay curve, but hard
// zero past the bleed distance — the CPU version's gradient stops end at
// t=1 with near-zero alpha, but never painted anything beyond that rect at
// all, whereas clamping t here without a hard cutoff left a constant
// residual glow (exp(-4) ~= 0.018) across the entire screen interior.
float edgeFalloff(float distFromEdge, float bleed) {
    if (bleed <= 0.0 || distFromEdge >= bleed) return 0.0;
    float t = distFromEdge / bleed;
    return exp(-EDGE_DECAY_RATE * t);
}

vec3 hueAt(vec2 uv, vec2 center, float offsetDeg) {
    vec2 d = uv - center;
    float angle = degrees(atan(d.y, d.x));
    float hueDeg = mod(angle + offsetDeg, 360.0);
    return hsv2rgb(hueDeg / 360.0, 0.65, 1.0);
}

// Color/alpha for a given state id, so we can evaluate both the "from" and
// "to" state and cross-fade between them — swapping u_state instantly (as
// before) meant the color hard-cut mid-dip instead of actually fading.
void stateColorAlpha(int state, vec2 uv, vec2 center, float edgeMix, out vec3 color, out float alpha) {
    if (state == 1) {
        // listening: angular rainbow — one hue per pixel based on its angle
        // around screen center. Averaging multiple hue passes (tried earlier)
        // samples the full spectrum locally and washes toward gray/white;
        // a single continuous hue field avoids that entirely.
        color = hueAt(uv, center, u_hueOffsetDeg);
        alpha = edgeMix * u_edgeAlphaListening + u_centerTint;
    } else if (state == 2) {
        color = u_colorAmber;
        alpha = edgeMix * u_pulseAlpha + u_centerTint;
    } else if (state == 3) {
        color = u_colorGreen;
        alpha = edgeMix * u_edgeAlphaSuccess + u_centerTint;
    } else if (state == 4) {
        color = u_colorRed;
        alpha = edgeMix * u_edgeAlphaUnrecognised + u_centerTint;
    } else {
        color = vec3(0.0);
        alpha = 0.0;
    }
}

void main() {
    vec2 uv = fragCoord; // 0..1, y-up (GL convention)
    vec2 center = vec2(0.5, 0.5);

    // Distance (0..1) from the nearest screen edge, per axis, in y-down terms
    // to match the CPU version's BLEED_TOP/LEFT (measured from top/left).
    float distTop = 1.0 - uv.y;     // 0 at top edge
    float distBottom = uv.y;        // 0 at bottom edge
    float distLeft = uv.x;          // 0 at left edge
    float distRight = 1.0 - uv.x;   // 0 at right edge

    float fTop = edgeFalloff(distTop, u_bleedTop);
    float fBottom = edgeFalloff(distBottom, u_bleedTop);
    float fLeft = edgeFalloff(distLeft, u_bleedLeft);
    float fRight = edgeFalloff(distRight, u_bleedLeft);
    float edgeMix = clamp(fTop + fBottom + fLeft + fRight, 0.0, 1.0);

    vec3 colorFrom, colorTo;
    float alphaFrom, alphaTo;
    stateColorAlpha(u_prevState, uv, center, edgeMix, colorFrom, alphaFrom);
    stateColorAlpha(u_state, uv, center, edgeMix, colorTo, alphaTo);

    vec3 color = mix(colorFrom, colorTo, u_crossfade);
    float alpha = mix(alphaFrom, alphaTo, u_crossfade);

    float finalAlpha = alpha * u_floodOpacity;
    // Premultiplied alpha: Windows layered windows (WS_EX_LAYERED) composite
    // expecting RGB already multiplied by A. Writing straight alpha here
    // made DWM composite the pill/flood as far more transparent than
    // intended — invisible live even though a raw framebuffer capture
    // (screenshot) still showed the un-composited straight-alpha content.
    outColor = vec4(color * finalAlpha, finalAlpha);
}
"""


class JarvisOverlayGL(QOpenGLWidget):
    def __init__(self):
        fmt = QSurfaceFormat()
        fmt.setAlphaBufferSize(8)
        fmt.setSamples(0)
        # Explicit swap interval: leaving this unset relies on the driver
        # default, which combined with a layered/transparent window can make
        # DWM throttle buffer swaps unpredictably and desync animation timing
        # from wall-clock time. 1 = sync to vblank (smooth, ~60fps typical).
        fmt.setSwapInterval(1)
        QSurfaceFormat.setDefaultFormat(fmt)

        super().__init__()

        self._state = "idle"
        self._prev_state = "idle"
        self._crossfade = 1.0  # 0 = fully _prev_state, 1 = fully _state
        self._transcript = ""
        self._flood_opacity = 0.0
        self._pill_reveal_radius = 0.0
        self._pill_x_offset = 0.0
        self._pulse_alpha = 0.0
        self._hue_offset = 0.0

        self._program = None
        self._vao = None

        self._setup_window()
        self._setup_animations()

    # --- window / GL setup ---

    def _setup_window(self):
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)

        screen = QGuiApplication.primaryScreen().geometry()
        self.setGeometry(screen)
        self.hide()

    def showEvent(self, event):
        super().showEvent(event)
        self._set_click_through(True)

    def _set_click_through(self, enabled: bool):
        if sys.platform != "win32":
            return
        import ctypes

        GWL_EXSTYLE = -20
        WS_EX_LAYERED = 0x00080000
        WS_EX_TRANSPARENT = 0x00000020

        hwnd = int(self.winId())
        user32 = ctypes.windll.user32
        style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
        if enabled:
            style |= WS_EX_LAYERED | WS_EX_TRANSPARENT
        else:
            style &= ~(WS_EX_LAYERED | WS_EX_TRANSPARENT)
        user32.SetWindowLongW(hwnd, GWL_EXSTYLE, style)

    def initializeGL(self):
        gl.glClearColor(0.0, 0.0, 0.0, 0.0)
        gl.glEnable(gl.GL_BLEND)
        # Premultiplied-alpha blend function, matching the shader's premultiplied output.
        gl.glBlendFunc(gl.GL_ONE, gl.GL_ONE_MINUS_SRC_ALPHA)

        fragment_src = FRAGMENT_SHADER.replace("__EDGE_DECAY_RATE__", repr(EDGE_DECAY_RATE))
        self._program = self._compile_program(VERTEX_SHADER, fragment_src)

        # Full-screen quad (two triangles), no VBO abstraction needed for 4 verts.
        vertices = [-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0]
        import array
        vertex_data = array.array('f', vertices)

        self._vao = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(self._vao)

        vbo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, vertex_data.tobytes(), gl.GL_STATIC_DRAW)
        gl.glVertexAttribPointer(0, 2, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
        gl.glEnableVertexAttribArray(0)

        gl.glBindVertexArray(0)

    @staticmethod
    def _compile_program(vertex_src: str, fragment_src: str):
        def compile_shader(src, shader_type):
            shader = gl.glCreateShader(shader_type)
            gl.glShaderSource(shader, src)
            gl.glCompileShader(shader)
            if not gl.glGetShaderiv(shader, gl.GL_COMPILE_STATUS):
                raise RuntimeError(gl.glGetShaderInfoLog(shader).decode())
            return shader

        vs = compile_shader(vertex_src, gl.GL_VERTEX_SHADER)
        fs = compile_shader(fragment_src, gl.GL_FRAGMENT_SHADER)

        program = gl.glCreateProgram()
        gl.glAttachShader(program, vs)
        gl.glAttachShader(program, fs)
        gl.glLinkProgram(program)
        if not gl.glGetProgramiv(program, gl.GL_LINK_STATUS):
            raise RuntimeError(gl.glGetProgramInfoLog(program).decode())

        gl.glDeleteShader(vs)
        gl.glDeleteShader(fs)
        return program

    def resizeGL(self, w, h):
        gl.glViewport(0, 0, w, h)

    def paintGL(self):
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        if self._flood_opacity > 0.0:
            gl.glUseProgram(self._program)

            def u(name):
                return gl.glGetUniformLocation(self._program, name)

            gl.glUniform2f(u("u_resolution"), float(self.width()), float(self.height()))
            gl.glUniform1f(u("u_floodOpacity"), self._flood_opacity)
            gl.glUniform1i(u("u_state"), STATE_IDS.get(self._state, 0))
            gl.glUniform1i(u("u_prevState"), STATE_IDS.get(self._prev_state, 0))
            gl.glUniform1f(u("u_crossfade"), self._crossfade)
            gl.glUniform1f(u("u_hueOffsetDeg"), self._hue_offset)
            gl.glUniform1f(u("u_pulseAlpha"), self._pulse_alpha)
            gl.glUniform1f(u("u_bleedTop"), BLEED_TOP)
            gl.glUniform1f(u("u_bleedLeft"), BLEED_LEFT)
            gl.glUniform1f(u("u_edgeAlphaListening"), EDGE_ALPHA_LISTENING)
            gl.glUniform1f(u("u_edgeAlphaSuccess"), EDGE_ALPHA_SUCCESS)
            gl.glUniform1f(u("u_edgeAlphaUnrecognised"), EDGE_ALPHA_UNRECOGNISED)
            gl.glUniform1f(u("u_centerTint"), CENTER_TINT_ALPHA)
            gl.glUniform3f(u("u_colorAmber"), *COLOR_AMBER)
            gl.glUniform3f(u("u_colorGreen"), *COLOR_GREEN)
            gl.glUniform3f(u("u_colorRed"), *COLOR_RED)

            gl.glBindVertexArray(self._vao)
            gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, 4)
            gl.glBindVertexArray(0)

        if self._pill_reveal_radius > 0.0:
            self._paint_pill_overlay()

    def _paint_pill_overlay(self):
        """Pill is small and text-heavy — drawn with QPainter over the GL
        surface rather than in the shader, since glyph rendering in GLSL
        isn't worth the complexity for a 240x30 box."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        px = self.width() - PILL_WIDTH - PILL_MARGIN_RIGHT + self._pill_x_offset
        py = PILL_MARGIN_TOP
        pill_rect = QRectF(px, py, PILL_WIDTH, PILL_HEIGHT)

        clip_path = QPainterPath()
        top_right = QPointF(px + PILL_WIDTH, py)
        clip_path.addEllipse(top_right, self._pill_reveal_radius, self._pill_reveal_radius)
        painter.setClipPath(clip_path)

        pill_path = QPainterPath()
        radius = PILL_HEIGHT / 2
        pill_path.addRoundedRect(pill_rect, radius, radius)

        painter.fillPath(pill_path, QColor(0, 0, 0, 255))
        painter.setPen(QColor(255, 255, 255, int(0.07 * 255)))
        painter.drawPath(pill_path)

        dot_r = 4
        dot_cx = pill_rect.left() + 10 + dot_r
        dot_cy = pill_rect.center().y()
        dot_color = QColor(*self._state_rgb_255())
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(dot_color)
        painter.drawEllipse(QPointF(dot_cx, dot_cy), dot_r, dot_r)

        text = self._transcript if self._state == "success" else STATE_TEXT.get(self._state, "")
        if text:
            painter.setPen(QColor(255, 255, 255))
            font = QFont("Segoe UI")
            font.setPixelSize(11)
            painter.setFont(font)
            text_left = dot_cx + dot_r + 8
            text_rect = QRectF(text_left, pill_rect.top(), pill_rect.right() - 12 - text_left, pill_rect.height())
            metrics = painter.fontMetrics()
            elided = metrics.elidedText(text, Qt.TextElideMode.ElideRight, int(text_rect.width()))
            painter.drawText(text_rect, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, elided)

        painter.end()

    def _state_rgb_255(self):
        import colorsys
        if self._state == "listening":
            h = (self._hue_offset % 360.0) / 360.0
            r, g, b = colorsys.hsv_to_rgb(h, 0.65, 1.0)
            return int(r * 255), int(g * 255), int(b * 255)
        if self._state == "processing":
            return tuple(int(c * 255) for c in COLOR_AMBER)
        if self._state == "success":
            return tuple(int(c * 255) for c in COLOR_GREEN)
        if self._state == "unrecognised":
            return tuple(int(c * 255) for c in COLOR_RED)
        return (255, 255, 255)

    # --- animations (identical structure/timing to overlay.py) ---

    def _setup_animations(self):
        self._appear_group = QParallelAnimationGroup(self)
        self._flood_in = QPropertyAnimation(self, b"floodOpacity")
        self._flood_in.setDuration(360)
        self._flood_in.setEasingCurve(QEasingCurve.Type.OutQuad)
        self._pill_in = QPropertyAnimation(self, b"pillRevealRadius")
        self._pill_in.setDuration(420)
        self._pill_in.setEasingCurve(QEasingCurve.Type.OutQuad)
        self._appear_group.addAnimation(self._flood_in)
        self._appear_group.addAnimation(self._pill_in)

        self._disappear_group = QParallelAnimationGroup(self)
        self._flood_out = QPropertyAnimation(self, b"floodOpacity")
        self._flood_out.setDuration(420)
        self._flood_out.setEasingCurve(QEasingCurve.Type.InOutQuad)
        self._pill_out = QPropertyAnimation(self, b"pillRevealRadius")
        self._pill_out.setDuration(380)
        self._pill_out.setEasingCurve(QEasingCurve.Type.InOutQuad)
        self._disappear_group.addAnimation(self._flood_out)
        self._disappear_group.addAnimation(self._pill_out)
        self._disappear_group.finished.connect(self._on_dismissed)

        self._pulse_anim = QPropertyAnimation(self, b"pulseAlpha")
        self._pulse_anim.setDuration(2000)
        self._pulse_anim.setEasingCurve(QEasingCurve.Type.InOutSine)
        self._pulse_anim.setLoopCount(-1)

        self._shake_anim = QPropertyAnimation(self, b"pillXOffset")
        self._shake_anim.setDuration(420)
        self._shake_anim.setEasingCurve(QEasingCurve.Type.InOutSine)
        self._shake_anim.setKeyValueAt(0.0, 0.0)
        self._shake_anim.setKeyValueAt(0.25, 5.0)
        self._shake_anim.setKeyValueAt(0.50, -5.0)
        self._shake_anim.setKeyValueAt(0.75, 5.0)
        self._shake_anim.setKeyValueAt(1.0, 0.0)

        self._hue_timer = QTimer(self)
        self._hue_timer.timeout.connect(self._advance_hue)

        self._dismiss_timer = QTimer(self)
        self._dismiss_timer.setSingleShot(True)
        self._dismiss_timer.timeout.connect(self._begin_disappear)

        # Cross-fade between states: animate crossfade 0->1 while the shader
        # blends colorFrom (_prev_state) and colorTo (_state) by that value,
        # plus a small opacity dip for a softer feel. Swapping u_state
        # instantly (the old approach) hard-cut the color mid-dip instead of
        # actually fading between the two.
        self._transition_anim = QPropertyAnimation(self, b"crossfade")
        self._transition_anim.setDuration(480)
        self._transition_anim.setEasingCurve(QEasingCurve.Type.InOutSine)

        self._transition_dip_anim = QPropertyAnimation(self, b"floodOpacity")
        self._transition_dip_anim.setDuration(480)
        self._transition_dip_anim.setEasingCurve(QEasingCurve.Type.InOutSine)

    # --- Q_PROPERTY-backed animated fields ---

    def _get_flood_opacity(self):
        return self._flood_opacity

    def _set_flood_opacity(self, value):
        self._flood_opacity = value
        self.update()

    floodOpacity = pyqtProperty(float, _get_flood_opacity, _set_flood_opacity)

    def _get_pill_reveal_radius(self):
        return self._pill_reveal_radius

    def _set_pill_reveal_radius(self, value):
        self._pill_reveal_radius = value
        self.update()

    pillRevealRadius = pyqtProperty(float, _get_pill_reveal_radius, _set_pill_reveal_radius)

    def _get_pill_x_offset(self):
        return self._pill_x_offset

    def _set_pill_x_offset(self, value):
        self._pill_x_offset = value
        self.update()

    pillXOffset = pyqtProperty(float, _get_pill_x_offset, _set_pill_x_offset)

    def _get_pulse_alpha(self):
        return self._pulse_alpha

    def _set_pulse_alpha(self, value):
        self._pulse_alpha = value
        self.update()

    pulseAlpha = pyqtProperty(float, _get_pulse_alpha, _set_pulse_alpha)

    def _get_crossfade(self):
        return self._crossfade

    def _set_crossfade(self, value):
        self._crossfade = value
        self.update()

    crossfade = pyqtProperty(float, _get_crossfade, _set_crossfade)

    # --- Public, thread-safe API ---

    def show_state(self, state: str, transcript: str = ""):
        QMetaObject.invokeMethod(
            self, "_apply_state", Qt.ConnectionType.QueuedConnection,
            Q_ARG(str, state), Q_ARG(str, transcript),
        )

    def cancel(self):
        """Dismiss whatever is currently showing, as if nothing happened —
        e.g. a false wake-word trigger. Just fades back to idle/hidden."""
        self.show_state("idle")

    # --- Internal state machine ---

    @pyqtSlot(str, str)
    def _apply_state(self, state: str, transcript: str):
        self._dismiss_timer.stop()
        self._transition_anim.stop()
        self._transition_dip_anim.stop()

        was_hidden = not self.isVisible()

        if state == "idle" and not was_hidden:
            # Fade out rather than instantly cutting to black — this is the
            # path show_state("idle") and cancel() both take.
            self._begin_disappear()
        elif was_hidden or state == "idle":
            self._prev_state = state
            self._crossfade = 1.0
            self._switch_to(state, transcript)
            if was_hidden and state != "idle":
                self.show()
                self._appear_group.stop()
                self._flood_in.setStartValue(0.0)
                self._flood_in.setEndValue(1.0)
                self._pill_in.setStartValue(0.0)
                self._pill_in.setEndValue(self._pill_diagonal())
                self._appear_group.start()
            else:
                self.update()
        else:
            # Snapshot the current on-screen state as the fade's "from" side,
            # switch the live state immediately (so its own animations start
            # on time), then crossfade the shader's blend from old to new.
            self._prev_state = self._state
            self._crossfade = 0.0
            self._switch_to(state, transcript)

            self._transition_anim.setStartValue(0.0)
            self._transition_anim.setEndValue(1.0)
            self._transition_anim.start()

            dip_depth = self._flood_opacity * 0.165  # ~30% of the old 0.55 dip
            self._transition_dip_anim.setStartValue(self._flood_opacity)
            self._transition_dip_anim.setKeyValueAt(0.5, self._flood_opacity - dip_depth)
            self._transition_dip_anim.setEndValue(self._flood_opacity)
            self._transition_dip_anim.start()

    def _switch_to(self, state: str, transcript: str):
        self._hue_timer.stop()
        self._pulse_anim.stop()
        self._shake_anim.stop()
        self._pill_x_offset = 0.0

        self._state = state
        self._transcript = transcript

        if state == "listening":
            self._hue_timer.start(33)
        elif state == "processing":
            self._pulse_anim.setStartValue(EDGE_ALPHA_PROCESSING_MIN)
            self._pulse_anim.setEndValue(EDGE_ALPHA_PROCESSING_MAX)
            self._pulse_anim.start()
        elif state == "unrecognised":
            self._shake_anim.start()
            self._dismiss_timer.start(AUTO_DISMISS_MS)
        elif state == "success":
            self._dismiss_timer.start(AUTO_DISMISS_MS)

        self.update()

    def _advance_hue(self):
        self._hue_offset = (self._hue_offset + 0.6) % 360.0
        self.update()

    def _pill_diagonal(self) -> float:
        return (PILL_WIDTH ** 2 + PILL_HEIGHT ** 2) ** 0.5

    def _begin_disappear(self):
        self._disappear_group.stop()
        self._flood_out.setStartValue(self._flood_opacity)
        self._flood_out.setEndValue(0.0)
        self._pill_out.setStartValue(self._pill_reveal_radius)
        self._pill_out.setEndValue(0.0)
        self._disappear_group.start()

    def _on_dismissed(self):
        self._hue_timer.stop()
        self._pulse_anim.stop()
        self._state = "idle"
        self._transcript = ""
        self.hide()


def _demo():
    app = QApplication(sys.argv)
    overlay = JarvisOverlayGL()

    sequence = [
        ("listening", "", 3000),
        ("processing", "", 2000),
        ("success", "open chrome", 2000),
        ("listening", "", 2000),
        ("processing", "", 2000),
        ("unrecognised", "", 2000),
    ]

    state = {"i": 0}

    def step():
        name, transcript, delay = sequence[state["i"]]
        overlay.show_state(name, transcript=transcript)
        state["i"] = (state["i"] + 1) % len(sequence)
        QTimer.singleShot(delay, step)

    QTimer.singleShot(300, step)
    sys.exit(app.exec())


if __name__ == "__main__":
    _demo()
