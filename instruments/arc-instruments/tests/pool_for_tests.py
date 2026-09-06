"""A registered-shaped pool for the command line tests, and nothing else.

The command line takes the study's item pool by module name (`--pool-module`), so a test of the
command line needs a module to name. This one stands in for a pool a person wrote for the study: it
is not marked smoke-only, so the confirmatory gate is allowed to accept it and the test reaches the
requirement it is actually about. It is not an instrument and no run should ever use it.
"""
from arc_runner import code_domain as CD


def pool(n: int = 12) -> CD.TaskPool:
    tasks = []
    for i in range(n):
        name = "add_%d" % i
        checks = ("assert %s(3, 4) == 7" % name, "assert %s(0, 0) == 0" % name,
                  "assert %s(-1, 1) == 0" % name)
        tasks.append(CD.Task(id=name, statement="Define %s(x, y) returning the sum." % name,
                             signature="def %s(x, y): ..." % name, shown_examples=(checks[0],),
                             checks=checks))
    return CD.TaskPool(tasks, name="pool-for-tests")
