# IrisMind agent instructions

- Keep every user query scoped through `request.user`; never expose another driver's records.
- Join driving data only with `user`, `session`, and timestamps. Never align datasets by row order.
- Keep risk weights and premium constants in `config/settings.py` and cover changes with tests.
- Never commit credentials, raw face media, private identifiers, or production datasets.
- Run `ruff check .` and `pytest` before proposing a commit.
- Do not commit, push, or open a pull request without explicit user approval.
