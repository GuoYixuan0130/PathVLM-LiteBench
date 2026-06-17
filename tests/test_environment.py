from pathvlm_litebench import version
from pathvlm_litebench.environment import collect_environment


def test_collect_environment_has_core_keys():
    env = collect_environment()
    assert env["pathvlm_litebench"] == version
    assert isinstance(env["python"], str) and env["python"]
    assert isinstance(env["platform"], str) and env["platform"]
    assert isinstance(env["packages"], dict)


def test_collect_environment_reports_installed_packages():
    packages = collect_environment()["packages"]
    # torch and numpy are hard runtime dependencies, so they must resolve.
    assert packages["torch"] is not None
    assert packages["numpy"] is not None


def test_collect_environment_is_json_serializable():
    import json

    json.dumps(collect_environment())
