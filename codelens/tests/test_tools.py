# -*- coding: utf-8 -*-
"""Unit tests for Web search and calculator tools."""

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import langchain_core.tools  # noqa: F401
except ModuleNotFoundError:
    langchain_core = types.ModuleType("langchain_core")
    tools_module = types.ModuleType("langchain_core.tools")

    def tool(fn=None, **_kwargs):
        if fn is None:
            return lambda f: f
        return fn

    tools_module.tool = tool
    langchain_core.tools = tools_module
    sys.modules["langchain_core"] = langchain_core
    sys.modules["langchain_core.tools"] = tools_module

from app.tools.calculator import calculate_expression  # noqa: E402
from app.tools.web_search import run_web_search  # noqa: E402


class CalculatorTests(unittest.TestCase):
    def test_basic_arithmetic(self):
        self.assertEqual(calculate_expression("2 + 3 * 4"), "14")

    def test_math_functions_and_constants(self):
        self.assertEqual(calculate_expression("sqrt(16) + sin(pi / 2)"), "5")

    def test_division_by_zero_returns_error(self):
        self.assertIn("division by zero", calculate_expression("1 / 0"))

    def test_import_is_rejected(self):
        result = calculate_expression('__import__("os").system("ls")')
        self.assertTrue(result.startswith("(error)"))

    def test_open_is_rejected(self):
        result = calculate_expression('open("/etc/passwd")')
        self.assertTrue(result.startswith("(error)"))
        self.assertIn("unsupported function", result)


class WebSearchTests(unittest.TestCase):
    def _fake_ddgs_module(self, ddgs_cls):
        module = types.ModuleType("ddgs")
        module.DDGS = ddgs_cls
        return module

    def test_formats_structured_results(self):
        class FakeDDGS:
            def text(self, query, max_results=5):
                self.query = query
                self.max_results = max_results
                return [
                    {
                        "title": "cpp-httplib releases",
                        "body": "Latest release information",
                        "href": "https://example.com/releases",
                    }
                ]

            def close(self):
                pass

        with patch.dict(sys.modules, {"ddgs": self._fake_ddgs_module(FakeDDGS)}):
            result = run_web_search("cpp-httplib latest release", max_results=3)

        self.assertIn("cpp-httplib releases", result)
        self.assertIn("Latest release information", result)
        self.assertIn("https://example.com/releases", result)

    def test_empty_results_are_clear(self):
        class EmptyDDGS:
            def text(self, query, max_results=5):
                return []

        with patch.dict(sys.modules, {"ddgs": self._fake_ddgs_module(EmptyDDGS)}):
            result = run_web_search("no such thing")

        self.assertIn("(no results)", result)

    def test_backend_errors_are_returned(self):
        class RaisingDDGS:
            def text(self, query, max_results=5):
                raise RuntimeError("network down")

        with patch.dict(sys.modules, {"ddgs": self._fake_ddgs_module(RaisingDDGS)}):
            result = run_web_search("cpp-httplib")

        self.assertIn("(error)", result)
        self.assertIn("network down", result)


if __name__ == "__main__":
    unittest.main()
