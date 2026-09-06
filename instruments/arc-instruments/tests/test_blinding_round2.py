from arc_instruments import blinding as bl


def test_marking_status_never_defaults_to_unmarked():
    assert bl.marking_status(True) == bl.KNOWN_MARKED
    assert bl.marking_status(False) == bl.KNOWN_UNMARKED
    assert bl.marking_status(None) == bl.UNKNOWN_MARKING


def test_transformation_preservation_gates_the_primary_endpoint():
    props = ["refusal", "negation", "commitment"]
    raw = [{"refusal": i % 3 == 0, "negation": i % 2 == 0, "commitment": "yes"} for i in range(400)]
    same = [dict(r) for r in raw]
    ok = bl.transformation_preservation(raw, same, props, floor=0.95)
    assert ok["admissible_as_primary"] is True
    flipped = [dict(r) for r in raw]
    for i in range(0, 400, 5):
        flipped[i]["refusal"] = not flipped[i]["refusal"]
    bad = bl.transformation_preservation(raw, flipped, props, floor=0.95)
    assert bad["admissible_as_primary"] is False
    assert bad["per_property"]["refusal"]["clears_floor"] is False
    assert bad["per_property"]["negation"]["clears_floor"] is True
    try:
        bl.transformation_preservation(raw, raw[:10], props)
    except ValueError:
        return
    raise AssertionError("unpaired items must be refused")
