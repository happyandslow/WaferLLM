# Color ID & Queue ID Valid Ranges

**Date**: 2026-02-18
**Context**: CSL color and queue resource allocation constraints
**Category**: gotcha

## Key Constraints

### Color IDs

| Type | Valid Range | Notes |
|---|---|---|
| Routable color | **1 ~ 20** | For fabric communication between PEs |
| Local task color | **8 ~ 20, 22 ~ 26** | For local task binding only (not routed) |

**Important**: Routable colors and local task colors share the same ID namespace on a single PE. They **cannot use the same ID** on the same PE. Plan your color allocation accordingly.

### Queue IDs

| Type | Valid Range | Notes |
|---|---|---|
| Input queue | **2 ~ 7** | Independent of color IDs |
| Output queue | **2 ~ 7** | Independent of color IDs |

Queue IDs are **not related to color IDs** — they are a separate resource. A color can be bound to any available queue via `@set_color_config` or DSD configuration.

## Allocation Example

```
PE color/queue allocation plan:
─────────────────────────────────
Routable colors:  1, 2, 3, 4      (fabric routing, IDs 1~20)
Local task colors: 8, 9, 10        (local only, IDs 8~20 or 22~26)
                                    ↑ no overlap with routable colors on same PE
Input queues:     2, 3, 4          (IDs 2~7, independent of colors)
Output queues:    2, 3             (IDs 2~7, independent of colors)
```

## Common Pitfalls

1. **Color ID collision**: Using the same ID for both a routable color and a local task color on the same PE — causes silent conflicts
2. **Queue ID out of range**: Attempting to use queue 0 or 1 — these are reserved by the system (memcpy)
3. **Confusing color IDs with queue IDs**: They are independent namespaces; color 3 and queue 3 are unrelated resources

## Related Resources

- `doc/csl/language/advanced-features.md` — color and routing configuration
- `doc/csl/language/task-ids.md` — task color binding
- `doc/csl/language/builtins.md` — `@get_color`, `@get_input_queue`, `@get_output_queue`
