"""Static validation of the Docker and Kubernetes build definitions.

These checks are cheap enough to run on every commit and catch the class of
mistake that otherwise only surfaces during a multi-minute image build, such as
passing a value to a boolean pip flag or referencing a manifest that does not
exist.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCKERFILE = REPO_ROOT / "Dockerfile"
COMPOSE = REPO_ROOT / "docker-compose.yml"
K8S = REPO_ROOT / "k8s"

# A hard import, not `pytest.importorskip`. A missing PyYAML previously made all
# 23 manifest checks skip, so CI reported green while validating nothing. This
# fails collection loudly instead, because pyyaml is a declared dev dependency.
try:
    import yaml
except ImportError as exc:  # pragma: no cover - environment problem
    raise RuntimeError(
        "PyYAML is required to validate the Docker and Kubernetes manifests. "
        "Install it with `pip install -r requirements-dev.txt`."
    ) from exc


@pytest.fixture(scope="module")
def dockerfile_text() -> str:
    return DOCKERFILE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def compose() -> dict:
    return yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))


def _k8s_docs() -> list[dict]:
    docs = []
    for path in sorted(K8S.glob("*.yaml")):
        docs.extend(d for d in yaml.safe_load_all(path.read_text(encoding="utf-8")) if d)
    return docs


class TestDockerfile:
    def test_uses_a_pinned_base_image(self, dockerfile_text):
        """Regression: `FROM python:3.12-slim` floats between rebuilds."""
        m = re.search(r"^ARG PYTHON_IMAGE=(\S+)$", dockerfile_text, re.M)
        assert m, "PYTHON_IMAGE build arg is not declared"
        base = m.group(1)
        assert "python:" in base, f"unexpected base image: {base}"
        # A fully pinned version, not a bare major.minor.
        assert re.search(r"python:\d+\.\d+\.\d+", base), f"base image not version-pinned: {base}"

        # Both stages must resolve the same ARG so they cannot drift.
        assert dockerfile_text.count("FROM ${PYTHON_IMAGE}") == 2
        assert "FROM python:" not in dockerfile_text, "a stage bypasses PYTHON_IMAGE"

    def test_installs_from_the_lock_file(self, dockerfile_text):
        assert "COPY requirements.txt requirements.lock" in dockerfile_text
        assert re.search(r"pip install[^\n]*-r /build/resolved\.txt", dockerfile_text)
        # The resolved file must be derived from the lock, not invented.
        assert re.search(r"cp /build/requirements\.lock /build/resolved\.txt", dockerfile_text)

    def test_boolean_pip_flags_take_no_value(self, dockerfile_text):
        """Regression: `--require-hashes=false` aborted the build.

        ``--require-hashes`` is a boolean switch, so pip exits with
        ``--require-hashes option does not take value`` and the image never
        builds. Docker's own flags (``--from=``, ``--chown=``) legitimately take
        values, so the check is scoped to the pip command lines.
        """
        pip_lines = [ln for ln in dockerfile_text.splitlines() if "pip install" in ln]
        assert pip_lines, "no pip install command found"
        for line in pip_lines:
            bad = re.findall(r"--[a-z][a-z-]*(?:hashes|quiet|no-cache|upgrade)=\S", line)
            assert not bad, f"boolean pip flag given a value: {bad}"
        assert "--require-hashes=false" not in dockerfile_text

    def test_runs_as_non_root(self, dockerfile_text):
        assert "USER ${APP_UID}:${APP_GID}" in dockerfile_text
        create_user = dockerfile_text.index("useradd")
        use_user = dockerfile_text.index("USER ")
        assert create_user < use_user, "USER must come after the account is created"

    def test_validates_lfs_pointers(self, dockerfile_text):
        assert "git-lfs.github.com/spec/v1" in dockerfile_text

    def test_has_a_healthcheck(self, dockerfile_text):
        assert "HEALTHCHECK" in dockerfile_text
        assert "_stcore/health" in dockerfile_text

    def test_provides_scratch_space_for_streamlit(self, dockerfile_text):
        """The root filesystem is read-only at runtime, so /tmp must exist."""
        assert "/tmp/streamlit" in dockerfile_text


class TestCompose:
    def test_service_runs_non_root_and_read_only(self, compose):
        service = compose["services"]["mri-app"]
        assert service.get("read_only") is True
        assert "no-new-privileges:true" in service.get("security_opt", [])

    def test_weights_are_mounted_read_only(self, compose):
        volumes = compose["services"]["mri-app"]["volumes"]
        weights = [v for v in volumes if "/app/weights" in v]
        assert weights, "weights volume is not mounted"
        assert weights[0].endswith(":ro"), f"weights should be read-only: {weights[0]}"

    def test_no_duplicate_mount_targets(self, compose):
        """Regression: a volume and a tmpfs on /tmp made compose fail to parse."""
        service = compose["services"]["mri-app"]
        targets = [v.split(":")[1] for v in service.get("volumes", [])]
        targets += [t.split(":")[0] for t in service.get("tmpfs", [])]
        assert len(targets) == len(set(targets)), f"duplicate mount targets: {targets}"

    def test_no_obsolete_version_key(self, compose):
        """`version` is obsolete in the Compose spec and warns on every command."""
        assert "version" not in compose

    def test_exposes_a_healthcheck(self, compose):
        assert "healthcheck" in compose["services"]["mri-app"]

    def test_terraform_free_build_arg_default(self, compose):
        args = compose["services"]["mri-app"]["build"].get("args", {})
        assert args.get("TENSORFLOW_DIST"), "TENSORFLOW_DIST build arg not exposed"


class TestKubernetesManifests:
    def test_all_manifests_parse(self):
        assert _k8s_docs()

    def test_required_manifests_present(self):
        names = {d.get("kind") for d in _k8s_docs()}
        assert {"Namespace", "ConfigMap", "Deployment", "Service"} <= names
        assert "NetworkPolicy" in names
        assert "PodDisruptionBudget" in names

    def test_image_tag_is_versioned(self):
        """Regression: `:latest` makes rollbacks impossible."""
        for doc in _k8s_docs():
            if doc.get("kind") != "Deployment":
                continue
            for container in doc["spec"]["template"]["spec"]["containers"]:
                image = container["image"]
                tag = image.rsplit(":", 1)[-1]
                assert tag != "latest", f"unpinned image tag: {image}"
                assert re.match(r"^v?\d+\.\d+", tag), f"tag is not a version: {image}"

    def test_pod_security_context_is_enforced(self):
        for doc in _k8s_docs():
            if doc.get("kind") != "Deployment":
                continue
            ctx = doc["spec"]["template"]["spec"]["securityContext"]
            assert ctx["runAsNonRoot"] is True
            assert ctx["runAsUser"] > 0
            assert "seccompProfile" in ctx

    def test_container_security_context(self):
        for doc in _k8s_docs():
            if doc.get("kind") != "Deployment":
                continue
            for container in doc["spec"]["template"]["spec"]["containers"]:
                ctx = container["securityContext"]
                assert ctx["readOnlyRootFilesystem"] is True
                assert ctx["allowPrivilegeEscalation"] is False
                assert ctx["capabilities"]["drop"] == ["ALL"]

    def test_read_only_root_has_scratch_volume(self):
        for doc in _k8s_docs():
            if doc.get("kind") != "Deployment":
                continue
            spec = doc["spec"]["template"]["spec"]
            mounts = {m["mountPath"] for c in spec["containers"] for m in c.get("volumeMounts", [])}
            volumes = {v["name"] for v in spec.get("volumes", [])}
            assert "/tmp" in mounts, "no /tmp mount for a read-only root filesystem"
            assert any(
                "emptyDir" in v for v in spec.get("volumes", []) if v["name"] in volumes
            )

    def test_probes_are_defined(self):
        for doc in _k8s_docs():
            if doc.get("kind") != "Deployment":
                continue
            for container in doc["spec"]["template"]["spec"]["containers"]:
                for probe in ("readinessProbe", "livenessProbe", "startupProbe"):
                    assert probe in container, f"missing {probe}"

    def test_resources_are_bounded(self):
        for doc in _k8s_docs():
            if doc.get("kind") != "Deployment":
                continue
            for container in doc["spec"]["template"]["spec"]["containers"]:
                limits = container["resources"]["limits"]
                assert "cpu" in limits and "memory" in limits

    def test_service_is_not_exposed_on_nodeports(self):
        """Regression: NodePort published an unauthenticated endpoint."""
        for doc in _k8s_docs():
            if doc.get("kind") != "Service":
                continue
            assert doc["spec"].get("type", "ClusterIP") == "ClusterIP"

    def test_network_policy_defaults_to_deny(self):
        policies = [d for d in _k8s_docs() if d.get("kind") == "NetworkPolicy"]
        assert policies, "no NetworkPolicy defined"
        assert any(
            set(p["spec"]["policyTypes"]) >= {"Ingress", "Egress"} and not p["spec"].get("ingress")
            and not p["spec"].get("egress")
            for p in policies
        ), "expected a default-deny policy"
