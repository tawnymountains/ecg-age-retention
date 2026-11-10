"""The package imports without scikit-survival and uses only its declared dependencies."""

import ast
import importlib.util
import os
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path

import ecg_age_retention

PACKAGE_DIR = Path(ecg_age_retention.__file__).resolve().parent
HAS_TORCH = importlib.util.find_spec("torch") is not None

# Runs in a fresh interpreter in which importing scikit-survival fails as if it were not installed.
BLOCK_SCIKIT_SURVIVAL = textwrap.dedent(
    """
    import importlib.abc
    import sys

    class NotInstalled(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name == "sksurv" or name.startswith("sksurv."):
                raise ModuleNotFoundError(f"No module named {name!r}")
            return None

    sys.meta_path.insert(0, NotInstalled())
    try:
        import sksurv
    except ModuleNotFoundError:
        pass
    else:
        raise AssertionError("scikit-survival was not blocked")
    """
)


def run_isolated(code):
    """Run ``code`` after blocking scikit-survival; return the completed process."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(PACKAGE_DIR.parent), env.get("PYTHONPATH", "")])
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        [sys.executable, "-B", "-c", BLOCK_SCIKIT_SURVIVAL + textwrap.dedent(code)],
        env=env,
        capture_output=True,
        text=True,
    )


class ImportWithoutScikitSurvivalTests(unittest.TestCase):
    def test_numpy_subpackages(self):
        result = run_isolated(
            """
            import ecg_age_retention
            import ecg_age_retention.data
            import ecg_age_retention.evaluation
            import ecg_age_retention.representation
            assert "sksurv" not in sys.modules
            assert "torch" not in sys.modules, "the NumPy subpackages must not import torch"
            print(ecg_age_retention.__version__)
            """
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "1.0.0")


@unittest.skipUnless(HAS_TORCH, "PyTorch is not installed")
class TorchImportWithoutScikitSurvivalTests(unittest.TestCase):
    def test_torch_subpackages(self):
        result = run_isolated(
            """
            import ecg_age_retention.models
            import ecg_age_retention.supervision
            assert "sksurv" not in sys.modules
            """
        )
        self.assertEqual(result.returncode, 0, result.stderr)


class DependencyTests(unittest.TestCase):
    def test_only_declared_dependencies_are_imported(self):
        allowed = {"numpy", "scipy", "sklearn", "torch"}
        for path in sorted(PACKAGE_DIR.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0:
                    names = [node.module]
                else:
                    continue
                for name in names:
                    top = name.split(".")[0]
                    with self.subTest(file=path.name, module=name):
                        self.assertTrue(
                            top in allowed or top in sys.stdlib_module_names or top == "__future__",
                            f"{path.name} imports {name}",
                        )


if __name__ == "__main__":
    unittest.main()
