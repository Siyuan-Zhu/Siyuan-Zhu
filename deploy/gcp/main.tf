terraform {
  required_version = ">= 1.6, < 2.0"
  required_providers {
    google = { source = "hashicorp/google", version = "~> 6.0" }
  }
}
provider "google" {
  project = var.project_id
  region  = var.region
}
variable "project_id" { default = "git-readme-features" }
variable "region" { default = "us-central1" }
variable "image_digest" {
  type = string
  validation {
    condition     = can(regex("@sha256:[a-f0-9]{64}$", var.image_digest))
    error_message = "Use an immutable reviewed container digest."
  }
}
# Resolve suitable existing identities/secrets from inventory before planning.
# No project-wide IAM changes, keys, secrets, APIs or billing changes here.
variable "runtime_service_account" { type = string }
variable "scheduler_service_account" { type = string }
variable "github_secret_id" { type = string }
variable "github_secret_version" { type = string }
variable "activate_schedules" { default = false }
resource "google_cloud_run_v2_job" "bridge" {
  for_each            = toset(["publish", "monitor"])
  name                = "zeitgeist-${each.key}"
  location            = var.region
  deletion_protection = true
  template {
    task_count  = 1
    parallelism = 1
    template {
      service_account = var.runtime_service_account
      max_retries     = 0
      timeout         = "1320s"
      containers {
        image = var.image_digest
        resources {
          limits = { cpu = "1", memory = "512Mi" }
        }
        env {
          name = "GITHUB_DISPATCH_TOKEN"
          value_source {
            secret_key_ref {
              secret  = var.github_secret_id
              version = var.github_secret_version
            }
          }
        }
        env {
          name  = "ZEITGEIST_MONITOR_ONLY"
          value = each.key == "monitor" ? "1" : "0"
        }
      }
    }
  }
}
resource "google_cloud_run_v2_job_iam_member" "invoke" {
  for_each = google_cloud_run_v2_job.bridge
  project  = var.project_id
  location = var.region
  name     = each.value.name
  role     = "roles/run.invoker"
  member   = "serviceAccount:${var.scheduler_service_account}"
}
resource "google_secret_manager_secret_iam_member" "token" {
  project   = var.project_id
  secret_id = var.github_secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${var.runtime_service_account}"
}
locals {
  schedules = {
    primary  = { cron = "0 2 * * *", job = "publish" }
    catchup  = { cron = "37 2,3 * * *", job = "publish" }
    deadline = { cron = "5 * * * *", job = "monitor" }
  }
}
resource "google_cloud_scheduler_job" "edition" {
  for_each         = local.schedules
  name             = "zeitgeist-${each.key}"
  region           = var.region
  schedule         = each.value.cron
  time_zone        = "America/New_York"
  paused           = !var.activate_schedules
  attempt_deadline = "180s"
  retry_config { retry_count = 0 }
  http_target {
    http_method = "POST"
    uri         = "https://run.googleapis.com/v2/projects/${var.project_id}/locations/${var.region}/jobs/${google_cloud_run_v2_job.bridge[each.value.job].name}:run"
    headers     = { "Content-Type" = "application/json" }
    body        = base64encode("{}")
    oauth_token { service_account_email = var.scheduler_service_account }
  }
}

variable "notification_channels" {
  type    = list(string)
  default = []
}
resource "google_logging_metric" "deadline" {
  name   = "zeitgeist_deadline_missing"
  filter = "resource.type=\"cloud_run_job\" AND jsonPayload.kind=\"deadline_observation\" AND jsonPayload.status=\"publication_missing\""
  metric_descriptor {
    metric_kind = "DELTA"
    value_type  = "INT64"
    unit        = "1"
  }
}
resource "google_monitoring_alert_policy" "deadline" {
  display_name          = "Zeitgeist publication missing after deadline"
  combiner              = "OR"
  notification_channels = var.notification_channels
  conditions {
    display_name = "Actual edition missing"
    condition_threshold {
      filter          = "metric.type=\"logging.googleapis.com/user/${google_logging_metric.deadline.name}\" AND resource.type=\"cloud_run_job\""
      comparison      = "COMPARISON_GT"
      threshold_value = 0
      duration        = "0s"
      aggregations {
        alignment_period   = "60s"
        per_series_aligner = "ALIGN_SUM"
      }
    }
  }
  lifecycle {
    precondition {
      condition     = !var.activate_schedules || length(var.notification_channels) > 0
      error_message = "Select an existing verified notification channel before cutover."
    }
  }
}
resource "google_logging_metric" "monitor_heartbeat" {
  name   = "zeitgeist_deadline_observed"
  filter = "resource.type=\"cloud_run_job\" AND jsonPayload.kind=\"deadline_observation\""
  metric_descriptor {
    metric_kind = "DELTA"
    value_type  = "INT64"
    unit        = "1"
  }
}
resource "google_monitoring_alert_policy" "monitor_missing" {
  display_name          = "Zeitgeist deadline monitor has not completed in 3 hours"
  combiner              = "OR"
  notification_channels = var.notification_channels
  conditions {
    display_name = "Deadline observation absent"
    condition_absent {
      filter   = "metric.type=\"logging.googleapis.com/user/${google_logging_metric.monitor_heartbeat.name}\" AND resource.type=\"cloud_run_job\""
      duration = "10800s"
      aggregations {
        alignment_period   = "3600s"
        per_series_aligner = "ALIGN_SUM"
      }
    }
  }
}
