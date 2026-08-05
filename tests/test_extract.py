from mint.helpers import extract


def test_torch_load_compat_omits_unsupported_weights_only(monkeypatch):
    observed = {}

    def fake_load(path, map_location=None, **kwargs):
        observed.update(path=path, map_location=map_location, kwargs=kwargs)
        return "checkpoint"

    monkeypatch.setattr(extract.torch, "load", fake_load)

    assert extract.torch_load_compat("model.ckpt", "cpu") == "checkpoint"
    assert observed == {"path": "model.ckpt", "map_location": "cpu", "kwargs": {}}


def test_torch_load_compat_disables_modern_weights_only_default(monkeypatch):
    observed = {}

    def fake_load(path, map_location=None, weights_only=True):
        observed.update(
            path=path,
            map_location=map_location,
            weights_only=weights_only,
        )
        return "checkpoint"

    monkeypatch.setattr(extract.torch, "load", fake_load)

    assert extract.torch_load_compat("model.ckpt", "cuda:0") == "checkpoint"
    assert observed == {
        "path": "model.ckpt",
        "map_location": "cuda:0",
        "weights_only": False,
    }


def test_collator_maximum_crop_start_preserves_requested_length(monkeypatch):
    collator = extract.CollateFn(truncation_seq_length=4)
    monkeypatch.setattr(extract.random, "randint", lambda lower, upper: upper)

    chains, chain_ids = collator([("AAAAA", "AAAAA")])

    assert tuple(chains.shape) == (1, 8)
    assert tuple(chain_ids.shape) == (1, 8)
