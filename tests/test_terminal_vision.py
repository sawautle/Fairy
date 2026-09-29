#!/usr/bin/env python3
"""
Tests for terminal image-path detection in fairy.py.

Tests:
  - _find_image_paths: pure function — finds image paths in text.
  - _expand_image_path_in_input: best-effort pipeline that prepends
    vision descriptions for local image files.
"""
from __future__ import annotations

import os
import tempfile
from unittest.mock import patch

import pytest

from fairy import _find_image_paths, _expand_image_path_in_input


# ── _find_image_paths (pure function) ────────────────────────────────────────

class TestFindImagePaths:
    """_find_image_paths scans text tokens for existing image files."""

    def test_no_image_paths_returns_empty(self):
        result = _find_image_paths("hello world")
        assert result == []

    def test_finds_existing_png(self):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            path = tmp.name
        try:
            result = _find_image_paths(f"look at {path}")
            assert result == [path]
        finally:
            os.unlink(path)

    def test_finds_existing_jpg(self):
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            path = tmp.name
        try:
            result = _find_image_paths(f"describe {path} for me")
            assert result == [path]
        finally:
            os.unlink(path)

    def test_finds_existing_jpeg(self):
        with tempfile.NamedTemporaryFile(suffix=".jpeg", delete=False) as tmp:
            path = tmp.name
        try:
            result = _find_image_paths(path)
            assert result == [path]
        finally:
            os.unlink(path)

    def test_finds_existing_webp(self):
        with tempfile.NamedTemporaryFile(suffix=".webp", delete=False) as tmp:
            path = tmp.name
        try:
            result = _find_image_paths(path)
            assert result == [path]
        finally:
            os.unlink(path)

    def test_finds_existing_gif(self):
        with tempfile.NamedTemporaryFile(suffix=".GIF", delete=False) as tmp:
            path = tmp.name
        try:
            # Case-insensitive extension
            result = _find_image_paths(f"animate {path}")
            assert result == [path]
        finally:
            os.unlink(path)

    def test_skips_nonexistent_paths(self):
        result = _find_image_paths(
            "C:\\nonexistent\\fake\\image.png C:\\also\\fake.jpeg"
        )
        assert result == []

    def test_skips_non_image_extensions(self):
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
            path = tmp.name
        try:
            result = _find_image_paths(f"read {path}")
            assert result == []
        finally:
            os.unlink(path)

    def test_multiple_image_paths(self):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp1:
            path1 = tmp1.name
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp2:
            path2 = tmp2.name
        try:
            result = _find_image_paths(f"{path1} and {path2}")
            assert len(result) == 2
            assert path1 in result
            assert path2 in result
        finally:
            os.unlink(path1)
            os.unlink(path2)


# ── _expand_image_path_in_input ────────────────────────────────────────────────

class TestExpandImagePathInInput:
    """_expand_image_path_in_input is best-effort — errors degrade gracefully."""

    def test_passthrough_when_no_image_paths(self):
        text = "Hello Fairy, how are you?"
        result = _expand_image_path_in_input(text)
        assert result == text

    def test_passthrough_when_vision_incapable(self):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            path = tmp.name
        try:
            with patch("controller.vision.is_vision_capable", return_value=False):
                result = _expand_image_path_in_input(f"look at {path}")
            # Should still return the text (with a note)
            assert "Hello" not in result  # text changed
            assert "Vision" in result or "doesn't support" in result
        finally:
            os.unlink(path)

    def test_prepends_description_for_existing_image(self):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            path = tmp.name
        try:
            with patch("controller.vision.is_vision_capable", return_value=True):
                with patch(
                    "controller.vision.describe_image",
                    return_value="It is a picture of a sunset over the ocean.",
                ):
                    result = _expand_image_path_in_input(f"what is in {path}")

            assert "It is a picture of a sunset" in result
            # The original text should still be present
            assert "what is in" in result
        finally:
            os.unlink(path)

    def test_multiple_images_concatenated(self):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp1:
            path1 = tmp1.name
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp2:
            path2 = tmp2.name
        try:
            with patch("controller.vision.is_vision_capable", return_value=True):
                with patch(
                    "controller.vision.describe_image",
                    side_effect=["Image 1 description.", "Image 2 description."],
                ):
                    result = _expand_image_path_in_input(f"compare {path1} and {path2}")

            assert "Image 1 description." in result
            assert "Image 2 description." in result
            assert "Image 1 of 2" in result
            assert "Image 2 of 2" in result
        finally:
            os.unlink(path1)
            os.unlink(path2)

    def test_nonexistent_file_not_crash(self):
        """A nonexistent file in the token list should be skipped gracefully."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            path = tmp.name
        try:
            # Mix real and fake paths
            fake = "C:\\nonexistent\\fake.png"
            with patch("controller.vision.is_vision_capable", return_value=True):
                with patch(
                    "controller.vision.describe_image",
                    return_value="Description.",
                ):
                    result = _expand_image_path_in_input(f"{path} and {fake}")

            # Should have description for real path, error note for fake
            assert "Description." in result
            assert "read failed" in result or fake in result
        finally:
            os.unlink(path)

    def test_describe_image_error_not_crash(self):
        """describe_image errors are caught and returned as strings."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            path = tmp.name
        try:
            with patch("controller.vision.is_vision_capable", return_value=True):
                with patch(
                    "controller.vision.describe_image",
                    return_value="[Vision] Something went wrong",
                ):
                    result = _expand_image_path_in_input(f"look at {path}")

            assert "Description." not in result
            assert "Something went wrong" in result
        finally:
            os.unlink(path)
