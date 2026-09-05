# Shared experiment code

Code imported by more than one experiment. Nothing here produces a result of its own,
and nothing here should be cited as evidence.

| File | What it does |
|------|--------------|
| `eden_gateway.py` | an OpenAI-compatible client wrapper used when an experiment is pointed at the shared gateway rather than at a provider directly. Standard library only, so it can be imported from any experiment directory without adding a dependency. |

## Configuration

The adapter reads two environment variables:

- `EDEN_GATEWAY_URL`
- `EDEN_GATEWAY_API_KEY`

An experiment that does not find them falls back to whatever provider credentials its own
`REPRODUCE.md` names. Neither variable is required to read this repository or to re-analyse
the results already recorded in it.
