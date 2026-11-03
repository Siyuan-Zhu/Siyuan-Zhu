# Zeitgeist scheduler migration — prepared, not provisioned

The preferred end state is Cloud Scheduler → Cloud Run Job running the existing
pipeline → canonical Firestore → GitHub profile publication. The first prepared
cutover is deliberately a **temporary bridge**: Scheduler → Cloud Run Job →
GitHub workflow_dispatch → existing shared GitHub writer queue. This preserves
serialization with visitor game/vote handlers while application leases are
commissioned. It still depends on GitHub runner availability and queue latency.
Do not describe it as an execution migration or a delivery SLA.

## Inventory and blocked prerequisites

The existing project is `git-readme-features`, default Firestore database in
`nam5`, with an existing GitHub WIF principal. Real read-only inspection on
November 3 returned Firestore HTTP 200. Listing enabled APIs, service accounts,
Cloud Run jobs, Scheduler locations and Secret Manager metadata returned HTTP
403. Therefore no claim is made that reusable resources do not exist.

Before any apply, an authorized operator must supply read-only inventory or
allow an inspection identity these permissions: `serviceusage.services.list`,
`iam.serviceAccounts.list`, `run.jobs.list`, Scheduler location/job listing,
and `secretmanager.secrets.list`. Do not grant those inspection rights to the
runtime merely for convenience. Reuse suitable existing resources; import any
matching resources into Terraform rather than replacing them.

Required plan inputs: immutable reviewed image digest, region, suitable existing
runtime and Scheduler service accounts, exact existing managed-secret ID/version,
and an existing verified Monitoring notification channel. The secret must contain
a GitHub credential limited to this repository (Actions read/write, Contents read,
Metadata read). Prefer a GitHub App installation token with a rotation mechanism;
a short-lived token without rotation is not unattended-ready. No provider keys
move for the bridge, and no service-account key is used.

## Prepared configuration

`main.tf` defines two Cloud Run Jobs (bridge and independent completion observer),
three Scheduler jobs, secret-level access, job-level invocation, a missing-edition
alert and a missing-monitor heartbeat alert. The hourly observer avoids the documented
23.5-hour maximum for metric-absence windows. All schedules default to paused.
No API enablement, project-wide roles, billing changes or secret values are in
this module. Build from the repository root with `deploy/gcp/Dockerfile`; deploy
an immutable image digest. Provider versions are locked.

The bridge dispatches once, correlates its unique execution ID to the GitHub run,
and verifies both the completed run and the actual public edition. An ambiguous
HTTP response never causes another dispatch. The deadline observer checks hourly, first enforcing the deadline at
04:05 Eastern whether that day's public edition exists; it reports an owner hold
separately. The 3-hour absence alert detects a missing monitor heartbeat after
its time series has first been established. Both alerts require live commissioning
and a verified notification destination before activation.

## Time and retry contract

Preserve `0 2 * * *` and `37 2,3 * * *` in America/New_York. These are one primary
and two catch-up opportunities, not three required generations. Each application
run checks canonical/public freshness and its persistent edition claim before
paid work. On spring-forward day the nonexistent 02:00/02:37 times may not run;
03:37 is the recovery opportunity. Fall-back does not create a second edition.
No date is derived from UTC midnight. Tests cover Eastern date boundaries;
actual DST-trigger delivery is not claimed as observed.

Cloud Run task retries and Scheduler retries are zero initially. Explicit
catch-ups and application reconciliation govern recovery. Idempotent public
pricing reads have three bounded attempts with 0.5/1-second backoff. Inference
transport timeouts retain their full financial reservation and stop; they do not
silently trigger another paid request. Expired claims during generation or
publication require reconciliation. An expired pre-inference claim is reclaimable.

## Incremental cost estimate (USD; not authorization or a bill cap)

Assume one 10-minute primary, two one-minute no-op catch-ups, and 24 one-minute
monitor runs per day: 66,960 seconds/month at one vCPU and 0.5 GiB. At instance-based
rates of 0.000018/vCPU-second and 0.000002/GiB-second, compute is approximately
1.27/month **before** free allowances. Three Scheduler jobs are 0.30/month before
the billing-account allowance; one secret version is about 0.06/month before its
allowance. Subtotal about **1.63/month**, plus artifact storage, egress, logging,
Monitoring and build usage. Shared free-tier availability is not established.
Long GitHub queues increase bridge runtime; 20-minute polling is bounded.
Existing provider spending and Firestore costs remain separate and unchanged.
This is a workload estimate, not a guarantee of zero or a new total budget cap.

Official references, checked November 3:
- https://cloud.google.com/run/pricing
- https://cloud.google.com/scheduler/pricing
- https://cloud.google.com/secret-manager/pricing
- https://docs.cloud.google.com/run/docs/execute/jobs-on-schedule
- https://docs.cloud.google.com/scheduler/docs/configuring/cron-job-schedules

## Cutover and verification

1. Finish inventory; select existing identities/secrets/notification channel.
2. Obtain approval for the concrete infrastructure plan and incremental costs.
3. Initialize and validate Terraform; plan with schedules paused. Review the
   exact resource/IAM diff and import any matching resources before apply.
4. Build/push the pinned image. In a private validation repository/namespace,
   validate duplicate dispatches, lost acknowledgements, failure reporting and
   content approval holds with no public writes. The current bridge is fixed to
   the production slug; a separate explicitly reviewed private target build is
   required for this cloud commissioning. Do not repurpose production secrets.
5. Verify notification delivery using a labeled test incident. Establish the
   heartbeat time series. Confirm a failed job and missing publication are
   distinguished from an accepted Scheduler invocation.
6. Disable GitHub cron via ZEITGEIST_SCHEDULE_ENABLED=false; leave editorial and
   visitor handlers enabled. Drain/check existing executions and claims.
7. Enable the GCP schedules. Do not run two independent production schedulers.
8. Observe a real scheduled execution through run completion and actual edition
   publication; inspect cost and duplicate prevention. Manual execution alone
   does not satisfy scheduled-delivery acceptance.

## Rollback and later direct execution

Pause GCP Scheduler first, inspect/drain Cloud Run and correlated GitHub jobs,
then restore the GitHub schedule switch. Never delete an ambiguous claim or
release uncertain financial holds. If canonical publication committed but Git
failed, reconcile the exact retained bundle with current visitor state; do not
regenerate or restore stale games/votes. Keep November 3 held until David approves
an exact publication identity, regardless of current date.

Direct Cloud Run execution remains a second cutover gate: use workload ADC,
managed provider secrets, durable private artifact storage, and a publication
lock shared with GitHub game/vote writers (or keep the final publisher in the
existing queue). Validate a full draft-only run before removing the bridge.
The current application still has GitHub/local credential and workspace coupling;
this preparation does not claim a direct Cloud Run pipeline is deployment-ready.
