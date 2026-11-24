# Plan-time tests with a mocked AWS provider: no AWS account or credentials needed.
# Run: tofu test   (or: terraform test, Terraform >= 1.7)

mock_provider "aws" {
  mock_data "aws_caller_identity" {
    defaults = { account_id = "123456789012" }
  }
  mock_data "aws_iam_policy_document" {
    defaults = { json = "{\"Version\":\"2012-10-17\",\"Statement\":[]}" }
  }
  mock_resource "aws_db_instance" {
    defaults = {
      address = "hcai-dev.example.eu-west-1.rds.amazonaws.com"
      port    = 5432
      arn     = "arn:aws:rds:eu-west-1:123456789012:db:hcai-dev"
    }
  }
  mock_resource "aws_iam_openid_connect_provider" {
    defaults = { arn = "arn:aws:iam::123456789012:oidc-provider/token.actions.githubusercontent.com" }
  }
  mock_resource "aws_ecs_cluster" {
    defaults = { arn = "arn:aws:ecs:eu-west-1:123456789012:cluster/hcai-dev" }
  }
  mock_resource "aws_iam_role" {
    defaults = { arn = "arn:aws:iam::123456789012:role/mock" }
  }
  mock_resource "aws_secretsmanager_secret" {
    defaults = { arn = "arn:aws:secretsmanager:eu-west-1:123456789012:secret:mock" }
  }
  mock_resource "aws_cloudwatch_log_group" {
    defaults = { arn = "arn:aws:logs:eu-west-1:123456789012:log-group:mock" }
  }
  mock_resource "aws_kms_key" {
    defaults = { arn = "arn:aws:kms:eu-west-1:123456789012:key/00000000-0000-0000-0000-000000000000" }
  }
  mock_resource "aws_sns_topic" {
    defaults = { arn = "arn:aws:sns:eu-west-1:123456789012:mock" }
  }
  mock_resource "aws_lb" {
    defaults = { arn = "arn:aws:elasticloadbalancing:eu-west-1:123456789012:loadbalancer/app/mock/abc", arn_suffix = "app/mock/abc", dns_name = "mock.eu-west-1.elb.amazonaws.com" }
  }
  mock_resource "aws_lb_target_group" {
    defaults = { arn = "arn:aws:elasticloadbalancing:eu-west-1:123456789012:targetgroup/mock/abc", arn_suffix = "targetgroup/mock/abc" }
  }
  mock_resource "aws_ecs_service" {
    defaults = { id = "arn:aws:ecs:eu-west-1:123456789012:service/hcai-dev/hcai-dev-api" }
  }
  mock_resource "aws_ecs_task_definition" {
    defaults = { arn = "arn:aws:ecs:eu-west-1:123456789012:task-definition/hcai-dev-api:1" }
  }
  mock_resource "aws_ecr_repository" {
    defaults = { repository_url = "123456789012.dkr.ecr.eu-west-1.amazonaws.com/hcai-dev-api", arn = "arn:aws:ecr:eu-west-1:123456789012:repository/hcai-dev-api" }
  }
}

variables {
  allowed_ingress_cidrs = ["203.0.113.10/32"]
  availability_zones    = ["eu-west-1a", "eu-west-1b"]
}

run "database_is_private_encrypted_and_tls_only" {
  command = apply

  assert {
    condition     = aws_db_instance.main.publicly_accessible == false
    error_message = "Database must not be publicly accessible"
  }
  assert {
    condition     = aws_db_instance.main.storage_encrypted
    error_message = "Database storage must be encrypted"
  }
  assert {
    condition     = aws_db_instance.main.manage_master_user_password
    error_message = "DB password must be managed by RDS/Secrets Manager, not Terraform"
  }
  assert {
    condition     = one([for p in aws_db_parameter_group.pg16.parameter : p.value if p.name == "rds.force_ssl"]) == "1"
    error_message = "TLS must be enforced on Postgres"
  }
  assert {
    condition     = aws_db_subnet_group.main.subnet_ids == toset(aws_subnet.private[*].id) && length(aws_subnet.private) == 2
    error_message = "Database must sit only in the two private subnets"
  }
}

run "network_exposure_is_minimal" {
  command = apply

  assert {
    condition     = length(aws_vpc_security_group_ingress_rule.alb_http) == 1 && aws_vpc_security_group_ingress_rule.alb_http["203.0.113.10/32"].cidr_ipv4 == "203.0.113.10/32"
    error_message = "Load balancer must only accept the allowed CIDRs"
  }
  assert {
    condition     = aws_vpc_security_group_ingress_rule.app_from_alb.referenced_security_group_id == aws_security_group.alb.id
    error_message = "API tasks must only accept traffic from the load balancer"
  }
  assert {
    condition     = aws_vpc_security_group_ingress_rule.db_from_app.referenced_security_group_id == aws_security_group.app.id
    error_message = "Database must only accept traffic from the API tasks"
  }
  assert {
    condition     = length(aws_route_table.private.route) == 0
    error_message = "Private (database) subnets must have no route to the internet"
  }
}

run "app_gets_secrets_not_plaintext" {
  command = apply

  assert {
    condition     = contains([for s in jsondecode(aws_ecs_task_definition.api.container_definitions)[0].secrets : s.name], "PGPASSWORD")
    error_message = "DB password must be injected from Secrets Manager"
  }
  assert {
    condition     = !contains([for e in jsondecode(aws_ecs_task_definition.api.container_definitions)[0].environment : e.name], "PGPASSWORD")
    error_message = "DB password must never be a plain environment variable"
  }
  assert {
    condition     = jsondecode(aws_ecs_task_definition.api.container_definitions)[0].readonlyRootFilesystem
    error_message = "Container root filesystem must be read-only"
  }
  assert {
    condition     = aws_ecs_service.api.deployment_circuit_breaker[0].rollback
    error_message = "Failed deployments must roll back automatically"
  }
}

run "bedrock_permissions_only_when_used" {
  command = apply

  assert {
    condition     = length(aws_iam_role_policy.task_bedrock) == 0
    error_message = "Task role must not get Bedrock access when llm_provider = anthropic"
  }
  assert {
    condition     = contains([for s in jsondecode(aws_ecs_task_definition.api.container_definitions)[0].secrets : s.name], "ANTHROPIC_API_KEY")
    error_message = "Claude API key must be injected when llm_provider = anthropic"
  }
}

run "bedrock_mode" {
  command = apply

  variables {
    llm_provider = "bedrock"
  }

  assert {
    condition     = length(aws_iam_role_policy.task_bedrock) == 1
    error_message = "Task role needs Bedrock access when llm_provider = bedrock"
  }
  assert {
    condition     = !contains([for s in jsondecode(aws_ecs_task_definition.api.container_definitions)[0].secrets : s.name], "ANTHROPIC_API_KEY")
    error_message = "No Claude API key should be injected in Bedrock mode"
  }
}

run "data_encrypted_with_customer_key" {
  command = apply

  assert {
    condition     = aws_cloudwatch_log_group.api.kms_key_id == aws_kms_key.main.arn && aws_secretsmanager_secret.anthropic_api_key.kms_key_id == aws_kms_key.main.arn
    error_message = "Logs and secrets must be encrypted with the customer-managed key"
  }
  assert {
    condition     = aws_kms_key.main.enable_key_rotation
    error_message = "KMS key rotation must be on"
  }
}

run "deploy_role_trusts_only_this_repo_environment" {
  command = apply

  assert {
    condition     = strcontains(aws_iam_role.github_deploy.assume_role_policy, "repo:pramodmakwana/healthcare-genai-platform:environment:dev")
    error_message = "Deploy role must only trust this repository's dev environment"
  }
}
