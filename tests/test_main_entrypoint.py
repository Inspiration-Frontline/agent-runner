import runpy


def test_direct_main_entrypoint_imports_without_starting_server() -> None:
    namespace = runpy.run_path("src/main.py", run_name="agent_runner_entrypoint_test")

    assert callable(namespace["create_app"])
