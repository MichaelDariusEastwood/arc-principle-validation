from models.policy_program import PolicyGenome
from training.verifier import verify_policy_source


def test_policy_compiles_and_returns_shape():
    src = PolicyGenome().render_source()
    res = verify_policy_source(src)
    assert res.ok, res.reason


def test_policy_rejects_imports():
    bad = "def policy(features):\n    import os\n    return {'lr_scale':1.0,'grad_clip':1.0,'noise_scale':0.0}\n"
    res = verify_policy_source(bad)
    assert not res.ok
    assert "Forbidden" in res.reason or "compile_error" in res.reason
