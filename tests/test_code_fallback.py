from controller import agent_controller


def test_cloud_coder_failure_uses_local_coder(monkeypatch):
    monkeypatch.setattr(agent_controller, "_OPENROUTER_AVAILABLE", True)
    monkeypatch.setattr(
        agent_controller,
        "ask_or_coder",
        lambda prompt: (_ for _ in ()).throw(RuntimeError("free models exhausted")),
    )
    monkeypatch.setattr(agent_controller.ollama_client, "unload", lambda model: None)
    monkeypatch.setattr(
        agent_controller.ollama_client,
        "chat",
        lambda **kwargs: {"message": {"content": "print(42)"}},
    )

    assert agent_controller._generate_raw_code("print 42", use_cloud=True) == "print(42)"


def test_code_sandbox_is_not_retried_by_outer_dispatch(monkeypatch):
    calls = []

    def failed_dispatch(name, arguments):
        calls.append((name, arguments))
        return {"ok": False, "error": "expected failure"}

    monkeypatch.setattr(agent_controller, "_dispatch_tool", failed_dispatch)
    agent_controller._run_tool("code_sandbox", {"description": "example"}, None)

    assert len(calls) == 1
