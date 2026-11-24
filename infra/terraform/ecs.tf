resource "aws_ecs_cluster" "main" {
  name = local.name

  setting {
    name  = "containerInsights"
    value = "enabled"
  }
}

resource "aws_cloudwatch_log_group" "api" {
  name              = "/ecs/${local.name}/api"
  retention_in_days = var.log_retention_days
  kms_key_id        = aws_kms_key.main.arn
}

locals {
  # Always populated because manage_master_user_password = true; try() only keeps the
  # mocked-provider tests (which can't mock this nested block on OpenTofu 1.8) working.
  db_secret_arn = try(aws_db_instance.main.master_user_secret[0].secret_arn, "arn:aws:secretsmanager:${var.region}:${local.account_id}:secret:pending")

  app_environment = [
    { name = "LLM_PROVIDER", value = var.llm_provider },
    { name = "LLM_MODEL", value = var.llm_model },
    { name = "LLM_FAST_MODEL", value = var.llm_fast_model },
    { name = "AWS_REGION", value = var.region },
    { name = "EMBEDDING_PROVIDER", value = "hashing" },
    { name = "VECTOR_STORE", value = "pgvector" },
    { name = "AGENT_CHECKPOINTER", value = "postgres" },
    { name = "PGHOST", value = aws_db_instance.main.address },
    { name = "PGPORT", value = tostring(aws_db_instance.main.port) },
    { name = "PGDATABASE", value = aws_db_instance.main.db_name },
    { name = "PGUSER", value = aws_db_instance.main.username },
    { name = "PGSSLMODE", value = "require" },
    { name = "LOG_FORMAT", value = "json" },
    { name = "SERVICE_NAME", value = "${local.name}-api" },
    { name = "METRICS_NAMESPACE", value = "HCAI" },
    { name = "REDACT_PHI_IN_LOGS", value = "true" },
  ]

  app_secrets = concat(
    [
      # RDS-managed secret is JSON {"username": ..., "password": ...}; inject just the password.
      { name = "PGPASSWORD", valueFrom = "${local.db_secret_arn}:password::" },
      { name = "ADMIN_TOKEN", valueFrom = aws_secretsmanager_secret.admin_token.arn },
    ],
    var.llm_provider == "anthropic" ? [{ name = "ANTHROPIC_API_KEY", valueFrom = aws_secretsmanager_secret.anthropic_api_key.arn }] : []
  )
}

resource "aws_ecs_task_definition" "api" {
  family                   = "${local.name}-api"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = var.task_cpu
  memory                   = var.task_memory
  execution_role_arn       = aws_iam_role.execution.arn
  task_role_arn            = aws_iam_role.task.arn

  runtime_platform {
    operating_system_family = "LINUX"
    cpu_architecture        = "X86_64"
  }

  volume {
    name = "tmp"
  }

  container_definitions = jsonencode([{
    name                   = "api"
    image                  = "${aws_ecr_repository.api.repository_url}:${var.image_tag}"
    essential              = true
    readonlyRootFilesystem = true
    user                   = "1000"
    portMappings           = [{ containerPort = 8000, protocol = "tcp" }]
    environment            = local.app_environment
    secrets                = local.app_secrets
    mountPoints            = [{ sourceVolume = "tmp", containerPath = "/tmp", readOnly = false }]
    healthCheck = {
      command     = ["CMD-SHELL", "python -c \"import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)\""]
      interval    = 30
      timeout     = 5
      retries     = 3
      startPeriod = 90
    }
    logConfiguration = {
      logDriver = "awslogs"
      options = {
        awslogs-group         = aws_cloudwatch_log_group.api.name
        awslogs-region        = var.region
        awslogs-stream-prefix = "api"
      }
    }
  }])
}

resource "aws_ecs_service" "api" {
  #checkov:skip=CKV_AWS_333:Public IP is outbound-only (inbound limited to the ALB security group); avoids ~USD 35/month NAT gateway. Production: private subnets + NAT/VPC endpoints.
  name                              = "${local.name}-api"
  cluster                           = aws_ecs_cluster.main.id
  task_definition                   = aws_ecs_task_definition.api.arn
  desired_count                     = var.desired_count
  launch_type                       = "FARGATE"
  health_check_grace_period_seconds = 120
  enable_execute_command            = false
  propagate_tags                    = "SERVICE"

  network_configuration {
    subnets          = aws_subnet.public[*].id
    security_groups  = [aws_security_group.app.id]
    assign_public_ip = true # outbound only; inbound is limited to the load balancer SG (no NAT cost)
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.api.arn
    container_name   = "api"
    container_port   = 8000
  }

  deployment_circuit_breaker {
    enable   = true
    rollback = true # failed deploys roll back automatically
  }

  deployment_minimum_healthy_percent = 100
  deployment_maximum_percent         = 200

  # GitHub Actions registers new task definition revisions on deploy; don't fight it.
  lifecycle {
    ignore_changes = [task_definition, desired_count]
  }

  depends_on = [aws_lb_listener.http]
}

resource "aws_appautoscaling_target" "api" {
  service_namespace  = "ecs"
  resource_id        = "service/${aws_ecs_cluster.main.name}/${aws_ecs_service.api.name}"
  scalable_dimension = "ecs:service:DesiredCount"
  min_capacity       = var.desired_count
  max_capacity       = var.max_count
}

resource "aws_appautoscaling_policy" "api_cpu" {
  name               = "${local.name}-cpu-target"
  policy_type        = "TargetTrackingScaling"
  service_namespace  = aws_appautoscaling_target.api.service_namespace
  resource_id        = aws_appautoscaling_target.api.resource_id
  scalable_dimension = aws_appautoscaling_target.api.scalable_dimension

  target_tracking_scaling_policy_configuration {
    target_value       = 60
    scale_in_cooldown  = 300
    scale_out_cooldown = 60
    predefined_metric_specification {
      predefined_metric_type = "ECSServiceAverageCPUUtilization"
    }
  }
}
