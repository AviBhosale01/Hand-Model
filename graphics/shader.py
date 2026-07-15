"""OpenGL 3.3 Core Profile shader program management.

Provides the ShaderProgram class for compiling, linking, and managing
GLSL shader programs with uniform caching and error reporting.
"""

import ctypes
import logging
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
from OpenGL.GL import (
    GL_COMPILE_STATUS,
    GL_FALSE,
    GL_FRAGMENT_SHADER,
    GL_INFO_LOG_LENGTH,
    GL_LINK_STATUS,
    GL_TRUE,
    GL_VERTEX_SHADER,
    glAttachShader,
    glCompileShader,
    glCreateProgram,
    glCreateShader,
    glDeleteProgram,
    glDeleteShader,
    glDetachShader,
    glGetProgramInfoLog,
    glGetProgramiv,
    glGetShaderInfoLog,
    glGetShaderiv,
    glGetUniformLocation,
    glLinkProgram,
    glShaderSource,
    glUniform1f,
    glUniform1i,
    glUniform3f,
    glUniform4f,
    glUniformMatrix4fv,
    glUseProgram,
)

logger = logging.getLogger(__name__)


class ShaderCompilationError(Exception):
    """Raised when a shader fails to compile."""
    pass


class ShaderLinkError(Exception):
    """Raised when a shader program fails to link."""
    pass


class ShaderProgram:
    """Manages an OpenGL shader program with vertex and fragment shaders.

    Compiles and links GLSL shaders, caches uniform locations for
    efficient repeated access, and provides typed uniform setters.

    Attributes:
        program_id: The OpenGL program object ID.
    """

    def __init__(self, vertex_path: str, fragment_path: str) -> None:
        """Create a shader program from vertex and fragment shader source files.

        Args:
            vertex_path: Filesystem path to the vertex shader GLSL source file.
            fragment_path: Filesystem path to the fragment shader GLSL source file.

        Raises:
            FileNotFoundError: If either shader source file does not exist.
            ShaderCompilationError: If either shader fails to compile.
            ShaderLinkError: If the program fails to link.
        """
        self._program_id: int = 0
        self._uniform_cache: Dict[str, int] = {}

        logger.info("Loading shader program: vert='%s', frag='%s'", vertex_path, fragment_path)

        # Read shader source files
        vertex_source = self._read_shader_file(vertex_path)
        fragment_source = self._read_shader_file(fragment_path)

        # Compile individual shaders
        vertex_shader = self._compile_shader(vertex_source, GL_VERTEX_SHADER, vertex_path)
        fragment_shader = self._compile_shader(fragment_source, GL_FRAGMENT_SHADER, fragment_path)

        # Link program
        self._program_id = self._link_program(vertex_shader, fragment_shader)

        # Clean up individual shader objects (they're linked into the program now)
        glDetachShader(self._program_id, vertex_shader)
        glDetachShader(self._program_id, fragment_shader)
        glDeleteShader(vertex_shader)
        glDeleteShader(fragment_shader)

        logger.info("Shader program created successfully (id=%d)", self._program_id)

    @staticmethod
    def _read_shader_file(filepath: str) -> str:
        """Read a shader source file from disk.

        Args:
            filepath: Path to the shader source file.

        Returns:
            The shader source code as a string.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        path = Path(filepath)
        if not path.is_file():
            raise FileNotFoundError(f"Shader source file not found: {filepath}")
        source = path.read_text(encoding='utf-8')
        logger.debug("Read shader source from '%s' (%d bytes)", filepath, len(source))
        return source

    @staticmethod
    def _compile_shader(source: str, shader_type: int, filepath: str = "<unknown>") -> int:
        """Compile a single GLSL shader.

        Args:
            source: The GLSL source code string.
            shader_type: GL_VERTEX_SHADER or GL_FRAGMENT_SHADER.
            filepath: Original file path for error reporting.

        Returns:
            The compiled shader object ID.

        Raises:
            ShaderCompilationError: If compilation fails.
        """
        type_name = "vertex" if shader_type == GL_VERTEX_SHADER else "fragment"
        shader = glCreateShader(shader_type)

        if shader == 0:
            raise ShaderCompilationError(
                f"Failed to create {type_name} shader object for '{filepath}'"
            )

        glShaderSource(shader, source)
        glCompileShader(shader)

        # Check compilation status
        status = glGetShaderiv(shader, GL_COMPILE_STATUS)
        if status != GL_TRUE:
            info_log = glGetShaderInfoLog(shader)
            if isinstance(info_log, bytes):
                info_log = info_log.decode('utf-8', errors='replace')

            # Log the error with source context
            logger.error(
                "Compilation failed for %s shader '%s':\n%s",
                type_name,
                filepath,
                info_log,
            )

            # Also log the source with line numbers for debugging
            numbered_lines = []
            for i, line in enumerate(source.splitlines(), start=1):
                numbered_lines.append(f"  {i:4d} | {line}")
            logger.debug(
                "Shader source for '%s':\n%s",
                filepath,
                "\n".join(numbered_lines),
            )

            glDeleteShader(shader)
            raise ShaderCompilationError(
                f"{type_name.capitalize()} shader compilation failed for '{filepath}':\n{info_log}"
            )

        logger.debug("Compiled %s shader '%s' successfully (id=%d)", type_name, filepath, shader)
        return shader

    @staticmethod
    def _link_program(vertex_shader: int, fragment_shader: int) -> int:
        """Link compiled vertex and fragment shaders into a program.

        Args:
            vertex_shader: Compiled vertex shader object ID.
            fragment_shader: Compiled fragment shader object ID.

        Returns:
            The linked program object ID.

        Raises:
            ShaderLinkError: If linking fails.
        """
        program = glCreateProgram()
        if program == 0:
            raise ShaderLinkError("Failed to create shader program object")

        glAttachShader(program, vertex_shader)
        glAttachShader(program, fragment_shader)
        glLinkProgram(program)

        # Check link status
        status = glGetProgramiv(program, GL_LINK_STATUS)
        if status != GL_TRUE:
            info_log = glGetProgramInfoLog(program)
            if isinstance(info_log, bytes):
                info_log = info_log.decode('utf-8', errors='replace')

            logger.error("Shader program linking failed:\n%s", info_log)
            glDeleteProgram(program)
            raise ShaderLinkError(f"Shader program linking failed:\n{info_log}")

        logger.debug("Linked shader program successfully (id=%d)", program)
        return program

    def use(self) -> None:
        """Activate this shader program for subsequent rendering calls."""
        glUseProgram(self._program_id)

    def _get_uniform_location(self, name: str) -> int:
        """Get the uniform location, using the cache for efficiency.

        Args:
            name: The uniform variable name in the shader.

        Returns:
            The uniform location integer. Returns -1 if the uniform
            is not found or has been optimized away.
        """
        if name in self._uniform_cache:
            return self._uniform_cache[name]

        location = glGetUniformLocation(self._program_id, name)
        if location == -1:
            logger.warning(
                "Uniform '%s' not found in shader program %d "
                "(may be optimized away or misspelled)",
                name,
                self._program_id,
            )
        self._uniform_cache[name] = location
        return location

    def set_mat4(self, name: str, matrix: np.ndarray) -> None:
        """Set a mat4 uniform.

        Args:
            name: Uniform name.
            matrix: A 4x4 numpy array (float32). Column-major order is assumed
                    (OpenGL default). The matrix is transposed if provided in
                    row-major (C-contiguous) order by numpy.
        """
        location = self._get_uniform_location(name)
        if location == -1:
            return
        mat = np.asarray(matrix, dtype=np.float32)
        glUniformMatrix4fv(location, 1, GL_FALSE, mat)

    def set_vec3(self, name: str, vector: Union[np.ndarray, tuple, list]) -> None:
        """Set a vec3 uniform.

        Args:
            name: Uniform name.
            vector: A 3-component vector as numpy array, tuple, or list.
        """
        location = self._get_uniform_location(name)
        if location == -1:
            return
        v = vector if not isinstance(vector, np.ndarray) else vector.flat
        glUniform3f(location, float(v[0]), float(v[1]), float(v[2]))

    def set_vec4(self, name: str, vector: Union[np.ndarray, tuple, list]) -> None:
        """Set a vec4 uniform.

        Args:
            name: Uniform name.
            vector: A 4-component vector as numpy array, tuple, or list.
        """
        location = self._get_uniform_location(name)
        if location == -1:
            return
        v = vector if not isinstance(vector, np.ndarray) else vector.flat
        glUniform4f(location, float(v[0]), float(v[1]), float(v[2]), float(v[3]))

    def set_float(self, name: str, value: float) -> None:
        """Set a float uniform.

        Args:
            name: Uniform name.
            value: The float value to set.
        """
        location = self._get_uniform_location(name)
        if location == -1:
            return
        glUniform1f(location, float(value))

    def set_int(self, name: str, value: int) -> None:
        """Set an int uniform (commonly used for sampler bindings).

        Args:
            name: Uniform name.
            value: The integer value to set.
        """
        location = self._get_uniform_location(name)
        if location == -1:
            return
        glUniform1i(location, int(value))

    def delete(self) -> None:
        """Delete the shader program and release GPU resources."""
        if self._program_id > 0:
            logger.debug("Deleting shader program (id=%d)", self._program_id)
            glDeleteProgram(self._program_id)
            self._program_id = 0
            self._uniform_cache.clear()

    @property
    def program_id(self) -> int:
        """The OpenGL program object ID."""
        return self._program_id

    def __del__(self) -> None:
        """Destructor hint — prefer calling delete() explicitly."""
        if self._program_id > 0:
            logger.warning(
                "ShaderProgram (id=%d) was garbage collected without explicit delete(). "
                "Call delete() to ensure proper GPU resource cleanup.",
                self._program_id,
            )

    def __repr__(self) -> str:
        return f"ShaderProgram(id={self._program_id}, cached_uniforms={len(self._uniform_cache)})"
