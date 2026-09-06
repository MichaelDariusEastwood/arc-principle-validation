"""An anchoring service for the command line tests, and nothing else.

`confirm` takes the operator's commitment service by module name (`--anchor-module`), so a test that
needs to reach PAST the anchoring requirement needs a module to name. This one returns the shape an
external service returns, exactly as the stub anchors inside the test files do: it is a stand-in for
a service somebody else runs, not a mock receipt, because a mock is refused on the deciding path and
a test that only ever saw the refusal could never show that the path beyond it exists.

It attests nothing to anybody and no run should ever use it.
"""
from arc_runner import custody


def anchor(sha256: str):
    return custody.receipt("stub-anchor:%s" % sha256[:12], sha256, service="test-stub")
