import logging
from typing import Tuple
from OpenGL.GL import *
from config.settings import Settings
from graphics.shader import ShaderProgram
from graphics.gl_utils import (
    create_fbo_with_texture,
    create_fullscreen_quad_vao,
    delete_fbo,
    delete_texture,
    delete_vao,
    delete_vbo
)

logger = logging.getLogger(__name__)

class BloomEffect:
    """Implements a multi-pass post-processing pipeline for Gaussian Bloom.
    Renders scene to high-precision HDR FBO -> extracts bright colors above threshold ->
    ping-pongs blur across separable horizontal/vertical passes -> overlays onto screen.
    """
    
    def __init__(self, width: int, height: int, settings: Settings):
        self._settings = settings
        self._width = width
        self._height = height
        
        # Framebuffer handles
        self._scene_fbo = 0
        self._scene_tex = 0
        self._depth_rbo = 0
        
        self._bright_fbo = 0
        self._bright_tex = 0
        
        self._ping_fbo = 0
        self._ping_tex = 0
        
        self._pong_fbo = 0
        self._pong_tex = 0
        
        self._quad_vao = 0
        self._quad_vbo = 0
        
        self._init_resources()

    def _init_resources(self) -> None:
        """Create textures, framebuffers, renderbuffers and fullscreen quad geometries."""
        # 1. Main Scene HDR Framebuffer
        self._scene_fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self._scene_fbo)
        
        self._scene_tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self._scene_tex)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, self._width, self._height, 0, GL_RGBA, GL_FLOAT, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self._scene_tex, 0)
        
        # Depth + Stencil Renderbuffer for depth testing inside FBO
        self._depth_rbo = glGenRenderbuffers(1)
        glBindRenderbuffer(GL_RENDERBUFFER, self._depth_rbo)
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, self._width, self._height)
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, self._depth_rbo)
        
        if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
            logger.error("Main Scene FBO is incomplete!")
            
        # 2. Bright Areas Extraction Framebuffer
        self._bright_fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self._bright_fbo)
        
        self._bright_tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self._bright_tex)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, self._width, self._height, 0, GL_RGBA, GL_FLOAT, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self._bright_tex, 0)
        
        if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
            logger.error("Bright FBO is incomplete!")

        # 3. Ping-Pong Framebuffers for Gaussian blur passes
        # Ping (Horizontal blur destination)
        self._ping_fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self._ping_fbo)
        self._ping_tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self._ping_tex)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, self._width, self._height, 0, GL_RGBA, GL_FLOAT, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self._ping_tex, 0)
        
        if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
            logger.error("Ping FBO is incomplete!")

        # Pong (Vertical blur destination)
        self._pong_fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self._pong_fbo)
        self._pong_tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self._pong_tex)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, self._width, self._height, 0, GL_RGBA, GL_FLOAT, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self._pong_tex, 0)
        
        if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
            logger.error("Pong FBO is incomplete!")

        # Unbind FBO
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        
        # 4. Initialize Fullscreen Quad geometry
        self._quad_vao, self._quad_vbo = create_fullscreen_quad_vao()

    def begin_scene_capture(self) -> None:
        """Directs all subsequent 3D render operations into the HDR framebuffer."""
        glBindFramebuffer(GL_FRAMEBUFFER, self._scene_fbo)
        glViewport(0, 0, self._width, self._height)
        # Clear color buffer and depth buffer
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    def end_scene_capture(self) -> None:
        """Restores rendering back to standard window framebuffer."""
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    def apply(
        self,
        shader_extract: ShaderProgram,
        shader_blur: ShaderProgram,
        shader_combine: ShaderProgram
    ) -> None:
        """Executes the bloom pipeline: bright pass, blur passes, and screen composition."""
        glDisable(GL_DEPTH_TEST)
        glDepthMask(GL_FALSE)
        glBindVertexArray(self._quad_vao)

        # 1. EXTRACT BRIGHT AREAS PASS
        glBindFramebuffer(GL_FRAMEBUFFER, self._bright_fbo)
        shader_extract.use()
        shader_extract.set_float("threshold", self._settings.bloom_threshold)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self._scene_tex)
        shader_extract.set_int("scene", 0)
        glDrawArrays(GL_TRIANGLES, 0, 6)

        # 2. SEPARABLE GAUSSIAN BLUR PASSES
        # Ping-pong between horizontal (ping_fbo) and vertical (pong_fbo)
        horizontal = True
        first_iteration = True
        passes = self._settings.bloom_blur_passes
        
        shader_blur.use()
        shader_blur.set_int("image", 0)
        
        for _ in range(passes * 2):
            target_fbo = self._ping_fbo if horizontal else self._pong_fbo
            glBindFramebuffer(GL_FRAMEBUFFER, target_fbo)
            
            shader_blur.set_int("horizontal", 1 if horizontal else 0)
            
            glActiveTexture(GL_TEXTURE0)
            if first_iteration:
                glBindTexture(GL_TEXTURE_2D, self._bright_tex)
                first_iteration = False
            else:
                glBindTexture(GL_TEXTURE_2D, self._pong_tex if horizontal else self._ping_tex)
                
            glDrawArrays(GL_TRIANGLES, 0, 6)
            horizontal = not horizontal

        # 3. COMBINE COMBOSITION PASS (Draw to default framebuffer 0)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        shader_combine.use()
        
        # Explicitly enable alpha blending to composite hologram onto camera feed
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self._scene_tex)
        shader_combine.set_int("scene", 0)
        
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, self._pong_tex)  # final blur result
        shader_combine.set_int("bloom_blur", 1)
        
        shader_combine.set_float("bloom_intensity", self._settings.bloom_intensity)
        
        glDrawArrays(GL_TRIANGLES, 0, 6)
        
        # Restore state
        glBindVertexArray(0)
        glEnable(GL_DEPTH_TEST)
        glDepthMask(GL_TRUE)

    def resize(self, width: int, height: int) -> None:
        """Re-creates the framebuffers when the window dimensions change."""
        self._width = width
        self._height = height
        self.cleanup()
        self._init_resources()

    def cleanup(self) -> None:
        """Deallocate all OpenGL resources."""
        if self._scene_fbo:
            glDeleteFramebuffers(1, [self._scene_fbo])
            self._scene_fbo = 0
        if self._scene_tex:
            glDeleteTextures(1, [self._scene_tex])
            self._scene_tex = 0
        if self._depth_rbo:
            glDeleteRenderbuffers(1, [self._depth_rbo])
            self._depth_rbo = 0
            
        if self._bright_fbo:
            glDeleteFramebuffers(1, [self._bright_fbo])
            self._bright_fbo = 0
        if self._bright_tex:
            glDeleteTextures(1, [self._bright_tex])
            self._bright_tex = 0
            
        if self._ping_fbo:
            glDeleteFramebuffers(1, [self._ping_fbo])
            self._ping_fbo = 0
        if self._ping_tex:
            glDeleteTextures(1, [self._ping_tex])
            self._ping_tex = 0
            
        if self._pong_fbo:
            glDeleteFramebuffers(1, [self._pong_fbo])
            self._pong_fbo = 0
        if self._pong_tex:
            glDeleteTextures(1, [self._pong_tex])
            self._pong_tex = 0
            
        if self._quad_vao:
            delete_vao(self._quad_vao)
            self._quad_vao = 0
        if self._quad_vbo:
            delete_vbo(self._quad_vbo)
            self._quad_vbo = 0
