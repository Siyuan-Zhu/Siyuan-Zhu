# Daily publishing and catch-up

The daily workflow targets 02:00 America/New_York. Two catch-up opportunities at
02:37 and 03:37 use the same serialized `readme-state-writer` group as editorial
and visitor publishing. GitHub may delay or drop scheduled events; these are
bounded recovery opportunities, not a guarantee of wall-clock delivery.

Before accounting maintenance or generation, each invocation verifies the local
publication's asset hashes and feed, then compares its identity and content with
canonical Firestore. A matching publication dated today is a successful no-op:
no provider calls, accounting maintenance or push. Manual daily dispatches obey
the same rule. Explicit owner editorial overrides remain separate.

A stale, coherent publication permits one normal pipeline attempt under the
existing spending and source-coverage controls. Publication failures, future
dates, corrupt assets, unavailable storage and canonical/public divergence stop
the job. An ambiguous outcome must be reconciled using its recovery evidence;
it must never be treated as permission to regenerate or restore old state.

Known Instagram `not_found` and `no_items` records are counted as unavailable
only when bound to an exact requested profile URL. `no_items` means empty or
private data, not proof that a profile is deleted. Other records and errors still
block collection; the existing fresh-source and tier-coverage gates still apply.
No posts are fabricated for unavailable profiles.

To recover a missed day, check running jobs first, run the owner-only read-only
preflight, inspect canonical/public consistency and any earlier provider outcome,
then dispatch the daily workflow once. A failed generation is not a reason to
blindly rerun it: inspect provider receipts and retained recovery artifacts first.
Keep the sandbox editorial writer and scheduler parked.

If all three GitHub schedule events are missed, this workflow cannot detect its
own absence. An independent freshness monitor would be a separate improvement.
