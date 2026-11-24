# Secret *containers* only. Values are set out-of-band so they never enter Terraform state:
#   aws secretsmanager put-secret-value --secret-id <arn> --secret-string 'sk-ant-...'

resource "aws_secretsmanager_secret" "anthropic_api_key" {
  name = "${local.name}/anthropic-api-key"
  #checkov:skip=CKV2_AWS_57:Third-party API key; rotate in the Anthropic console, then put-secret-value.
  description             = "Claude API key (only used when llm_provider = anthropic)"
  kms_key_id              = aws_kms_key.main.arn
  recovery_window_in_days = 7
}

resource "aws_secretsmanager_secret" "admin_token" {
  name = "${local.name}/admin-token"
  #checkov:skip=CKV2_AWS_57:Static admin token for a demo endpoint; rotate manually (runbook).
  description             = "Token protecting POST /ingest"
  kms_key_id              = aws_kms_key.main.arn
  recovery_window_in_days = 7
}
