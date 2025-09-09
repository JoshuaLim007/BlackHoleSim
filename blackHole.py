# pip install moderngl glfw numpy imageio
# Run: python raymarch_blackhole_volumetric_accumulate_doppler_kerr_rotating_disk.py

import numpy as np
import glfw
import moderngl
import math
import imageio
import time
import threading
import os

# ---------------- Window & Context ----------------
if not glfw.init():
    raise SystemExit("Failed to initialize GLFW")

glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
WIDTH, HEIGHT = 1280, 720
window = glfw.create_window(WIDTH, HEIGHT, "Raymarch Kerr Black Hole with Rotating Disk", None, None)
if not window:
    glfw.terminate()
    raise SystemExit("Failed to create GLFW window")

glfw.make_context_current(window)
ctx = moderngl.create_context()

glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_DISABLED)

# ---------------- Mouse Callback ----------------
yaw, pitch = 0.0, 0.0
last_x, last_y = WIDTH / 2, HEIGHT / 2
first_mouse = True

def mouse_callback(win, xpos, ypos):
    global yaw, pitch, last_x, last_y, first_mouse
    if first_mouse:
        last_x, last_y = xpos, ypos
        first_mouse = False

    xoffset = xpos - last_x
    yoffset = last_y - ypos
    last_x, last_y = xpos, ypos

    sensitivity = 0.1
    xoffset *= sensitivity
    yoffset *= sensitivity

    yaw += xoffset
    pitch += yoffset
    pitch = max(-89.0, min(89.0, pitch))

glfw.set_cursor_pos_callback(window, mouse_callback)

# ---------------- Load HDR Panorama Texture ----------------
pano_image = imageio.imread("pan1.jpg")[:, :, :3]  # Ensure it's RGB
pano_image = np.flipud(pano_image).astype('u1')  # Flip vertically and ensure 8-bit per channel
pano_texture = ctx.texture(pano_image.shape[1::-1], 3, pano_image.tobytes())  # Create RGB texture
pano_texture.build_mipmaps()
pano_texture.use(location=0)

# ---------------- Fullscreen Triangle ----------------
quad = ctx.buffer(np.array([
    -1.0, -1.0,
     3.0, -1.0,
    -1.0,  3.0,
], dtype='f4').tobytes())

def on_file_change(file_path):
    print(f"File {file_path} has changed. Reloading shaders...")
    global vertex_shader, fragment_shader, prog, vao

    try:
        vertex_shader_t = open(shader_files[0], 'r').read()
        fragment_shader_t = open(shader_files[1], 'r').read()
        prog_t = ctx.program(vertex_shader=vertex_shader_t, fragment_shader=fragment_shader_t)
        vao_t = ctx.vertex_array(prog_t, [(quad, '2f', 'in_pos')])
        prog_t['panoTex'].value = 0
    except Exception as e:
        print(f"Error reloading shaders: {e}")
        return
    
    #free existing resources
    vao.release()
    prog.release()

    vertex_shader = vertex_shader_t
    fragment_shader = fragment_shader_t
    prog = prog_t
    vao = vao_t

shader_files = ['blackhole.vert', 'blackhole.frag']
last_modified_time = []
last_modified_time.append(os.path.getmtime(shader_files[0]))
last_modified_time.append(os.path.getmtime(shader_files[1]))

vertex_shader = open(shader_files[0], 'r').read()
fragment_shader = open(shader_files[1], 'r').read()
prog = ctx.program(vertex_shader=vertex_shader, fragment_shader=fragment_shader)
vao = ctx.vertex_array(prog, [(quad, '2f', 'in_pos')])
prog['panoTex'].value = 0

# ---------------- Camera ----------------
camPos = np.array([0.0, 0.0, 5.0], dtype=np.float32)
movement_speed = 5.0
start_time = time.time()
current_modified_time = []
current_modified_time.append(last_modified_time[0])
current_modified_time.append(last_modified_time[1])

while not glfw.window_should_close(window):
    
    current_modified_time[0] = os.path.getmtime(shader_files[0])
    current_modified_time[1] = os.path.getmtime(shader_files[1])
    if current_modified_time != last_modified_time:
        last_modified_time[0] = current_modified_time[0]
        last_modified_time[1] = current_modified_time[1]
        on_file_change(shader_files)

    fbw, fbh = glfw.get_framebuffer_size(window)
    ctx.viewport = (0, 0, fbw, fbh)
    ctx.clear(0.0, 0.0, 0.0)

    rad_yaw = math.radians(yaw)
    rad_pitch = math.radians(pitch)
    forward_x = math.cos(rad_pitch) * math.cos(rad_yaw)
    forward_y = math.sin(rad_pitch)
    forward_z = math.cos(rad_pitch) * math.sin(rad_yaw)
    forward = np.array([forward_x, forward_y, forward_z], dtype=np.float32)

    right = np.cross(forward, np.array([0.0, 1.0, 0.0], dtype=np.float32))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)

    dt = 0.016
    velocity = movement_speed * dt
    if glfw.get_key(window, glfw.KEY_W) == glfw.PRESS:
        camPos += forward * velocity
    if glfw.get_key(window, glfw.KEY_S) == glfw.PRESS:
        camPos -= forward * velocity
    if glfw.get_key(window, glfw.KEY_A) == glfw.PRESS:
        camPos -= right * velocity
    if glfw.get_key(window, glfw.KEY_D) == glfw.PRESS:
        camPos += right * velocity
    if glfw.get_key(window, glfw.KEY_SPACE) == glfw.PRESS:
        camPos += up * velocity
    if glfw.get_key(window, glfw.KEY_LEFT_CONTROL) == glfw.PRESS:
        camPos -= up * velocity

    prog['iResolution'].value = (float(fbw), float(fbh))
    prog['camDir'].value = forward
    prog['camPos'].value = tuple(camPos)
    prog['iTime'].value = 5; #float(time.time() - start_time)

    vao.render(moderngl.TRIANGLES)
    glfw.swap_buffers(window)
    glfw.poll_events()

glfw.terminate()