# Alarms -> SNS (email), a dashboard built on the app's EMF metrics plus AWS service metrics,
# and a saved Logs Insights query to trace one request end-to-end by request_id.

locals {
  service_dim = "${local.name}-api"
  ns          = "HCAI"
}

resource "aws_sns_topic" "alarms" {
  name              = "${local.name}-alarms"
  kms_master_key_id = "alias/aws/sns"
}

resource "aws_sns_topic_subscription" "email" {
  count     = var.alarm_email == "" ? 0 : 1
  topic_arn = aws_sns_topic.alarms.arn
  protocol  = "email"
  endpoint  = var.alarm_email
}

locals {
  alarms = {
    api-5xx = {
      description = "API returning server errors"
      namespace   = local.ns
      metric      = "Errors"
      dimensions  = { Service = local.service_dim, Endpoint = "/ask" }
      stat        = "Sum"
      threshold   = 5
      period      = 300
    }
    ask-latency-p95 = {
      description = "p95 /ask latency above 8s"
      namespace   = local.ns
      metric      = "LatencyMs"
      dimensions  = { Service = local.service_dim, Endpoint = "/ask" }
      stat        = "p95"
      threshold   = 8000
      period      = 300
    }
    low-groundedness = {
      description = "Answers failing the groundedness check (possible hallucination or retrieval regression)"
      namespace   = local.ns
      metric      = "LowGroundedness"
      dimensions  = { Service = local.service_dim, Endpoint = "/ask:rag" }
      stat        = "Sum"
      threshold   = 5
      period      = 900
    }
    guardrail-spike = {
      description = "Unusual number of blocked prompts (abuse or prompt-injection attempts)"
      namespace   = local.ns
      metric      = "GuardrailBlocked"
      dimensions  = { Service = local.service_dim, Endpoint = "/ask:rag" }
      stat        = "Sum"
      threshold   = 20
      period      = 900
    }
    llm-cost-hourly = {
      description = "LLM spend above $2/hour"
      namespace   = local.ns
      metric      = "CostUSD"
      dimensions  = { Service = local.service_dim, Endpoint = "/ask:rag" }
      stat        = "Sum"
      threshold   = 2
      period      = 3600
    }
    rds-cpu = {
      description = "Database CPU above 80%"
      namespace   = "AWS/RDS"
      metric      = "CPUUtilization"
      dimensions  = { DBInstanceIdentifier = aws_db_instance.main.identifier }
      stat        = "Average"
      threshold   = 80
      period      = 300
    }
    alb-unhealthy-targets = {
      description = "No healthy API tasks behind the load balancer"
      namespace   = "AWS/ApplicationELB"
      metric      = "UnHealthyHostCount"
      dimensions  = { LoadBalancer = aws_lb.api.arn_suffix, TargetGroup = aws_lb_target_group.api.arn_suffix }
      stat        = "Maximum"
      threshold   = 0
      period      = 60
    }
  }
}

resource "aws_cloudwatch_metric_alarm" "this" {
  for_each            = local.alarms
  alarm_name          = "${local.name}-${each.key}"
  alarm_description   = each.value.description
  namespace           = each.value.namespace
  metric_name         = each.value.metric
  dimensions          = each.value.dimensions
  statistic           = contains(["Sum", "Average", "Maximum", "Minimum", "SampleCount"], each.value.stat) ? each.value.stat : null
  extended_statistic  = contains(["Sum", "Average", "Maximum", "Minimum", "SampleCount"], each.value.stat) ? null : each.value.stat
  period              = each.value.period
  evaluation_periods  = 1
  threshold           = each.value.threshold
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.alarms.arn]
  ok_actions          = [aws_sns_topic.alarms.arn]
}

resource "aws_cloudwatch_dashboard" "main" {
  dashboard_name = local.name
  dashboard_body = jsonencode({
    widgets = [
      {
        type = "metric", x = 0, y = 0, width = 12, height = 6
        properties = {
          title = "Requests and errors", region = var.region, view = "timeSeries", stat = "Sum", period = 60
          metrics = [
            [local.ns, "Requests", "Service", local.service_dim, "Endpoint", "/ask"],
            [local.ns, "Errors", "Service", local.service_dim, "Endpoint", "/ask"],
            [local.ns, "AgentRuns", "Service", local.service_dim, "Endpoint", "agent:prior_auth"],
            [local.ns, "AgentRuns", "Service", local.service_dim, "Endpoint", "agent:discharge"],
          ]
        }
      },
      {
        type = "metric", x = 12, y = 0, width = 12, height = 6
        properties = {
          title = "/ask latency (ms)", region = var.region, view = "timeSeries", period = 60
          metrics = [
            [local.ns, "LatencyMs", "Service", local.service_dim, "Endpoint", "/ask", { stat = "p50", label = "p50" }],
            [local.ns, "LatencyMs", "Service", local.service_dim, "Endpoint", "/ask", { stat = "p95", label = "p95" }],
            [local.ns, "RetrievalMs", "Service", local.service_dim, "Endpoint", "/ask:rag", { stat = "p95", label = "retrieval p95" }],
          ]
        }
      },
      {
        type = "metric", x = 0, y = 6, width = 12, height = 6
        properties = {
          title = "Answer quality and safety", region = var.region, view = "timeSeries", stat = "Sum", period = 300
          metrics = [
            [local.ns, "LowGroundedness", "Service", local.service_dim, "Endpoint", "/ask:rag"],
            [local.ns, "Abstained", "Service", local.service_dim, "Endpoint", "/ask:rag"],
            [local.ns, "GuardrailBlocked", "Service", local.service_dim, "Endpoint", "/ask:rag"],
          ]
        }
      },
      {
        type = "metric", x = 12, y = 6, width = 12, height = 6
        properties = {
          title = "LLM tokens and cost", region = var.region, view = "timeSeries", stat = "Sum", period = 300
          metrics = [
            [local.ns, "InputTokens", "Service", local.service_dim, "Endpoint", "/ask:rag"],
            [local.ns, "OutputTokens", "Service", local.service_dim, "Endpoint", "/ask:rag"],
            [local.ns, "CostUSD", "Service", local.service_dim, "Endpoint", "/ask:rag", { yAxis = "right" }],
          ]
        }
      },
      {
        type = "metric", x = 0, y = 12, width = 12, height = 6
        properties = {
          title = "ECS service CPU / memory (%)", region = var.region, view = "timeSeries", stat = "Average", period = 60
          metrics = [
            ["AWS/ECS", "CPUUtilization", "ClusterName", aws_ecs_cluster.main.name, "ServiceName", aws_ecs_service.api.name],
            ["AWS/ECS", "MemoryUtilization", "ClusterName", aws_ecs_cluster.main.name, "ServiceName", aws_ecs_service.api.name],
          ]
        }
      },
      {
        type = "metric", x = 12, y = 12, width = 12, height = 6
        properties = {
          title = "Database", region = var.region, view = "timeSeries", stat = "Average", period = 60
          metrics = [
            ["AWS/RDS", "CPUUtilization", "DBInstanceIdentifier", aws_db_instance.main.identifier],
            ["AWS/RDS", "DatabaseConnections", "DBInstanceIdentifier", aws_db_instance.main.identifier, { yAxis = "right" }],
          ]
        }
      },
    ]
  })
}

resource "aws_cloudwatch_query_definition" "trace_request" {
  name            = "${local.name}/trace-request"
  log_group_names = [aws_cloudwatch_log_group.api.name]
  query_string    = <<-EOT
    fields @timestamp, level, logger, msg, Endpoint, LatencyMs, InputTokens, CostUSD, guardrail, model
    | filter request_id = "REPLACE_WITH_REQUEST_ID"
    | sort @timestamp asc
  EOT
}

resource "aws_cloudwatch_query_definition" "slow_requests" {
  name            = "${local.name}/slowest-requests"
  log_group_names = [aws_cloudwatch_log_group.api.name]
  query_string    = <<-EOT
    filter Endpoint = "/ask" and ispresent(LatencyMs)
    | stats max(LatencyMs) as worst_ms by request_id
    | sort worst_ms desc
    | limit 20
  EOT
}
