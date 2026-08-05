from __future__ import annotations

import json
from pathlib import Path
import tomllib
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ReleaseSurfaceTests(unittest.TestCase):
    def test_package_includes_public_platform(self) -> None:
        project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        self.assertEqual(project["project"]["requires-python"], "==3.13.5")
        self.assertEqual(project["project"]["license"], "Apache-2.0")
        dependencies = project["project"]["dependencies"]
        for prefix in ("fastapi", "httpx", "modal", "psycopg", "torch", "uvicorn"):
            with self.subTest(dependency=prefix):
                self.assertTrue(any(item.startswith(prefix) for item in dependencies))
        self.assertEqual(
            project["tool"]["setuptools"]["packages"],
            [
                "benchmark",
                "benchmark.manifests",
                "client",
                "data",
                "research",
                "service",
                "service.static",
            ],
        )
        self.assertEqual(
            project["tool"]["setuptools"]["package-data"]["service"],
            ["static/*.css", "static/*.svg"],
        )

    def test_platform_source_and_deployment_files_are_present(self) -> None:
        for relative in (
            "service/app.py",
            "service/db.py",
            "service/evaluator.py",
            "service/tiers.py",
            "modal_runner.py",
            "Dockerfile",
            "Dockerfile.modal-deploy",
            "docker-compose.yml",
            ".env.example",
            "scripts/deploy_modal.sh",
            "scripts/generate_datasets.sh",
        ):
            with self.subTest(path=relative):
                self.assertTrue((ROOT / relative).is_file())

    def test_backup_artifacts_are_absent(self) -> None:
        backups = [
            path.relative_to(ROOT)
            for path in ROOT.rglob("*.orig")
            if ".git" not in path.parts and ".venv" not in path.parts
        ]
        self.assertEqual(backups, [])

    def test_generator_covers_public_accelerator_manifests_only(self) -> None:
        generator = (ROOT / "scripts" / "generate_datasets.sh").read_text(
            encoding="utf-8"
        )
        manifest_dir = ROOT / "benchmark" / "manifests"
        public_manifests = sorted(manifest_dir.glob("h100_easy_*.json")) + sorted(
            manifest_dir.glob("h100_medium_*.json")
        )
        self.assertEqual(len(public_manifests), 10)
        for manifest_path in public_manifests:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            data_root = manifest["data"]["data_root"]
            with self.subTest(manifest=manifest_path.name):
                self.assertIn(f"--output_dir {data_root}", generator)

        generation_commands = [
            line for line in generator.splitlines() if line.startswith("python -m data.")
        ]
        self.assertEqual(len(generation_commands), 10)
        self.assertTrue(
            all(line.startswith("python -m data.squaring_mod") for line in generation_commands)
        )
        self.assertEqual(list(manifest_dir.glob("h100_hard_*.json")), [])
        self.assertNotIn("\n# Hard:", generator)

    def test_open_source_documents_are_present(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertTrue((ROOT / "LICENSE").is_file())
        self.assertIn("one-layer submit", readme)
        self.assertIn("uv sync", readme)
        self.assertIn("Apache License 2.0", readme)


if __name__ == "__main__":
    unittest.main()
