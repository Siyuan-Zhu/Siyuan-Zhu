# Daily refresh recovery

The existing GitHub writer queue serializes editorial publication, moves and votes.
The primary opportunity is 02:00 America/New_York, with 02:37 and 03:37 catch-ups
and a 04:00 freshness target. Scheduled events can arrive late. A green job or
accepted trigger alone does not prove a timely public publication.

## Automatic recovery limits

- Current accepted edition: verify its canonical/public identity and do no generation.
- Source outage: reuse real retained evidence only within the existing source,
  tier, diversity, provenance and freshness rules. Orbit publication age remains
  at most 336 hours, Macro 96 hours, and Instagram fetch age 72 hours. Future
  timestamps are invalid. Cached model vectors must satisfy existing model checks.
- Required coverage missing: stop before embeddings, keep the last complete digest
  with its original date, and record `source_collection` plus bounded health counts.
- Anonymous pricing GETs: at most three requests per fixed URL within a 55-second
  retry budget; each HTTP phase has at most 15 seconds. There is no hard aggregate
  wall-clock cancellation. Honor Retry-After, add bounded jitter, and wait at most
  five seconds inline. Longer valid hints defer to an existing catch-up. Invalid
  hints, auth failures, changed prices/units and unapproved redirects stop.
- A later same-source claim is permitted only for the local maintenance CLI's
  exact exit-75 transient classification after zero attempted accounting writes.
  It waits for the 20-minute claim lease and Retry-After, with at most three claims
  per edition across existing opportunities. Catch-ups are not three generations.
- Paid calls, accounting mutations and uncertain acknowledgements receive no new
  automatic retry. Existing provider polling observes the same request identity.
- Git publication uses the existing bounded non-force retry and rereads current
  canonical visitor state. Missing/mismatched artwork or changed runtime source
  stops rather than invoking generation or restoring stale state.

## Stop and alert

Stop for unknown paid outcomes, attempted/ambiguous accounting writes, incomplete
or corrupt bundles, missing credentials, deterministic pricing/configuration
failures, missing required sources, quality rejection, approval holds, or exhausted
attempt limits. Retain reservations, receipts, candidate bytes and the original
edition date. Daily attempt receipts distinguish generation from Git publication
and include fixed recovery guidance. Independent monitoring checks public
freshness and its own health; notification infrastructure is separately managed.

A pending `data/publication_reconcile.json` blocks new generation. Inspect its
publication identity and retained bundle, then read canonical state and actual
Git. A known canonical commit with failed Git needs publication reconciliation
against the latest votes/game/history, never a new digest. Unknown commit or
download-tracking acknowledgement must be resolved before another mutation.

## Owner decisions and rollback

Changes to editorial/source/freshness/quality rules, paid-call replay, financial
settings, schedules, permissions or infrastructure require a concrete review.
Issue comments and email replies execute no recovery commands. Failed workers
cannot revive generation through a state transition.

Rollback code with a reviewed non-force successor after draining affected writers.
Keep current accepted assets, canonical activity, accounting and recovery journals.
Do not reset the repository, restore a stale database, clear claims, or rerun
generation merely to repair presentation or publication. Inspect uncertain
outcomes before retrying. Verify the actual public identity and assets afterward.
