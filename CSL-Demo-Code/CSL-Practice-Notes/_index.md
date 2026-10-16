# CSL Practice Notes

> **Purpose**: Staging area for practical findings discovered while writing CSL code.
> Once enough notes accumulate, consolidate them into `AI-Knowledge-Base/` for permanent use.

## Workflow

1. **Record**: Write down notes as you encounter them during actual CSL development
2. **Accumulate**: Let notes build up over time across different topics
3. **Consolidate**: Periodically review and merge mature notes into `AI-Knowledge-Base/`
4. **Clean up**: Archive or remove merged notes from this folder

## Suggested File Naming

Use topic-based filenames with a clear prefix:

| Prefix | Category | Example |
|---|---|---|
| `gotcha-` | Pitfalls & common mistakes | `gotcha-queue-init-wse3.md` |
| `tip-` | Practical tips & tricks | `tip-dsd-reuse-pattern.md` |
| `opt-` | Performance optimizations | `opt-fifo-depth-tuning.md` |
| `debug-` | Debugging techniques | `debug-trace-buffer-usage.md` |
| `pattern-` | Reusable code patterns | `pattern-state-machine.md` |
| `compat-` | WSE-2/WSE-3 compatibility | `compat-multi-fabric-input.md` |

## Note Template

```markdown
# [Title]

**Date**: YYYY-MM-DD
**Context**: [What you were working on when you discovered this]
**Category**: gotcha | tip | opt | debug | pattern | compat

## Problem / Observation

[What happened or what you noticed]

## Solution / Insight

[What you learned or how you solved it]

## Code Example

```csl
// Relevant code snippet
```

## Related Resources

- [Links to relevant docs, examples, or AI-Knowledge-Base entries]
```

## Consolidation Status

| Topic | Notes Count | Merged to KB | Date |
|---|---|---|---|
| Operator Fusion & Reconfig | 2 | No | 2026-03-17 |

## Notes by Topic

### Operator Fusion (2026-03-17)
- `pattern-operator-fusion-reconfig.md` — Full pattern: shared_tile, color reuse, queue rebinding, shared module instantiation
- `gotcha-operator-fusion-pitfalls.md` — 5 gotchas: missing unblock, double instantiation, queue conflicts, stale data, layout vs runtime routes
- `gotcha-gemm-output-column-major.md` — GEMM output is column-major due to save_address ordering; causes layout mismatch with GEMV input
