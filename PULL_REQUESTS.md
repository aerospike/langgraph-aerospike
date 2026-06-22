# Pull request guide

This repo enforces PR title format in CI and provides a body template on GitHub. Use this guide when opening a pull request.

## PR title (required — CI enforced)

Every PR title is validated by the [PR Hygiene workflow](.github/workflows/pr-hygiene.yml) using [Conventional Commits](https://www.conventionalcommits.org/) and, for some types, a Jira ticket.

### Format

```text
type(scope): [JIRA-123] short description
```

| Part | Required | Rules |
|------|----------|-------|
| `type` | Yes | Lowercase conventional commit type (see table below) |
| `(scope)` | No | Lowercase; letters, digits, `-`, `_` (e.g. `cookbooks`, `checkpoint`, `ci`) |
| `[JIRA-123]` | Sometimes | Uppercase project key + hyphen + number; required for `feat` and `fix` |
| `description` | Yes | Imperative, concise summary (no trailing period) |

### Valid `type` values

These are validated by `@commitlint/config-conventional`:

| Type | When to use |
|------|-------------|
| `feat` | New feature or user-facing capability |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `refactor` | Code change that neither fixes a bug nor adds a feature |
| `perf` | Performance improvement |
| `test` | Adding or updating tests |
| `build` | Build system or external dependencies |
| `ci` | CI configuration or scripts |
| `chore` | Maintenance (deps, tooling) that doesn't fit other types |
| `revert` | Reverts a previous commit |

### Jira ticket rules

- **Required** when `type` is `feat` or `fix`
- **Optional** for all other types (`docs`, `chore`, `ci`, etc.)
- Ticket must appear immediately after the colon, in square brackets, before the description
- Pattern: `[A-Z]{2,10}-[0-9]+` (e.g. `[LANG-42]`, `[INFRA-370]`)

### Title examples

```text
docs(cookbooks): add fork-from-checkpoint cookbook
fix(checkpoint): [LANG-99] handle missing parent checkpoint on fork
docs: document PR title conventions
chore(deps): refresh uv.lock via weekly upgrade
ci: run deptry on both packages
refactor(store): simplify namespace key encoding
test(checkpoint): cover TTL expiration edge case
```

Invalid examples:

```text
Add fork cookbook                          # not conventional commit format
feat: add fork cookbook                    # feat requires [JIRA-123]
feat(cookbooks): add fork cookbook         # feat requires [JIRA-123]
Feat(cookbooks): [LANG-42] add fork        # type must be lowercase
feat(Cookbooks): [LANG-42] add fork       # scope must be lowercase
feat(cookbooks): LANG-42 add fork         # ticket must be in [brackets]
```

## PR body

GitHub pre-fills the description from [`.github/PULL_REQUEST_TEMPLATE.md`](.github/PULL_REQUEST_TEMPLATE.md). Fill in each section:

| Section | What to include |
|---------|-----------------|
| **Description** | Brief overview of the problem and approach |
| **Changes Made** | Bullet list of main code/doc changes |
| **Testing** | Commands run, scenarios verified (e.g. `uv run pytest`, manual steps) |
| **Screenshots** | UI or output captures, if relevant |
| **Additional Notes** | Breaking changes, follow-ups, reviewer callouts |

Link related GitHub issues when applicable.
