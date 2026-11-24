# Runbook: AWS deployment

How to deploy, operate and tear down the platform on AWS. Everything is in `infra/terraform` (works with Terraform ≥ 1.7 or OpenTofu ≥ 1.8).

> **Data:** this stack is for **synthetic data only**. Real patient data needs an AWS account covered by the right agreements (GDPR DPA; HIPAA BAA if US data), HTTPS, WAF, private subnets, Multi-AZ, deletion protection and a DPIA. See "Before real data" at the end.

## What gets created

| Component | Resource | Why |
|---|---|---|
| Network | VPC, 2 public + 2 private subnets, no NAT gateway | DB in private subnets with no internet route; NAT omitted to save ~$35/month |
| API | ECS Fargate service (0.5 vCPU / 1 GB) behind an ALB, CPU autoscaling 1→3 | Serverless containers; ALB only accepts your IP |
| Database | RDS PostgreSQL 16 (db.t4g.micro) + pgvector | Vector index **and** LangGraph agent checkpoints |
| Secrets | Secrets Manager: RDS-managed DB password, Claude API key, admin token | No secrets in code, Terraform state or GitHub |
| Encryption | One customer-managed KMS key (rotation on) | Logs, secrets, Performance Insights |
| Images | ECR, immutable tags (git SHA), scan on push | Every deploy traceable to a commit |
| Observability | CloudWatch dashboard, 7 alarms → SNS email, Logs Insights saved queries | See "Monitoring" |
| CI/CD | GitHub OIDC role (main branch, `dev` environment only) | No long-lived AWS keys |

**Approximate cost (eu-west-1, running 24/7): USD 65–85/month.** The main items are the ALB (~$20), Fargate (~$18), RDS + storage (~$16), public IPv4 addresses (~$11), and CloudWatch custom metrics, alarms and Container Insights (~$8–12). Claude API usage is billed separately by Anthropic. Check with the [AWS Pricing Calculator](https://calculator.aws/) before you apply. **Tear down when you're not demoing** (last section).

## 1. One-time setup

Run these from WSL (`user@host:~$`), not PowerShell. You need the AWS CLI logged in to your own account (`aws sts get-caller-identity`) and Terraform or OpenTofu.

```bash
cd infra/terraform
cp environments/dev.tfvars.example environments/dev.tfvars
# edit: allowed_ingress_cidrs = ["<your IP>/32"]   (curl -s https://checkip.amazonaws.com)
#       alarm_email = "you@example.com"

# Remote state (recommended): create a versioned, encrypted bucket once, then
cp backend.hcl.example backend.hcl   # edit bucket name
terraform init -backend-config=backend.hcl
# (or for a quick personal sandbox: delete the backend "s3" {} block and use local state)
```

## 2. First apply

The service needs an image before it can start. Apply in two steps:

```bash
# a) Registry first
terraform apply -var-file=environments/dev.tfvars -target=aws_ecr_repository.api

# b) Push a bootstrap image (run from the repo root)
REPO=$(terraform -chdir=infra/terraform output -raw ecr_repository_url)
aws ecr get-login-password --region eu-west-1 | docker login --username AWS --password-stdin ${REPO%/*}
docker build -t $REPO:bootstrap . && docker push $REPO:bootstrap

# c) Everything else (~10 minutes; RDS takes longest)
terraform -chdir=infra/terraform apply -var-file=environments/dev.tfvars
```

Then set the secret values. They're set out-of-band so they never enter Terraform state:

```bash
aws secretsmanager put-secret-value --secret-id "$(terraform output -raw anthropic_secret_arn)" --secret-string 'sk-ant-...'
aws secretsmanager put-secret-value --secret-id "$(terraform output -raw admin_token_secret_arn)" --secret-string "$(openssl rand -hex 24)"
aws ecs update-service --cluster "$(terraform output -raw ecs_cluster)" --service "$(terraform output -raw ecs_service)" --force-new-deployment
```

Confirm the SNS subscription email if you set `alarm_email`.

**Bedrock instead of the Claude API** (keeps prompts inside AWS): enable Anthropic model access in the Bedrock console for eu-west-1. Then set `llm_provider = "bedrock"`, and set `llm_model` / `llm_fast_model` to the model or inference-profile IDs shown in the console. The task role gets `bedrock:InvokeModel` for Anthropic models only.

## 3. Connect GitHub Actions (continuous deployment)

In the GitHub repo, go to **Settings → Environments → New environment `dev`** (optionally require a reviewer). Add these **variables** (not secrets) from `terraform output`:

| Variable | Value |
|---|---|
| `AWS_REGION` | `eu-west-1` |
| `AWS_DEPLOY_ROLE_ARN` | `github_deploy_role_arn` |
| `ECR_REPOSITORY_URL` | `ecr_repository_url` |
| `ECS_CLUSTER` / `ECS_SERVICE` | `ecs_cluster` / `ecs_service` |
| `TASK_DEFINITION_FAMILY` | `task_definition_family` |
| `APP_SUBNET_IDS` | `app_subnet_ids` joined with commas |
| `APP_SECURITY_GROUP_ID` | `app_security_group_id` |
| `API_URL` | `api_url` |

From then on, every push to `main` that passes CI runs `deploy.yml`: build → push `:<sha>` → register task definition → **ingestion task** → rolling update → smoke test. If the new tasks fail health checks, the ECS circuit breaker rolls back automatically.

`infra.yml` runs `fmt`, `validate`, mocked-provider tests (`tofu test`) and Checkov on every infrastructure change. It needs no AWS access. `apply` stays a deliberate human step.

## Monitoring

- **Dashboard:** `terraform output dashboard_url`. Shows requests and errors, /ask p50/p95 latency, retrieval latency, groundedness failures, abstentions, guardrail blocks, tokens, LLM cost, ECS and RDS health.
- **Alarms (email):** 5xx errors, p95 latency > 8s, low-groundedness answers > 5/15min, guardrail-block spike, LLM cost > $2/hour, RDS CPU > 80%, no healthy targets.
- **Tracing one request:** every response has an `X-Request-ID` header (send your own to choose it). In CloudWatch Logs Insights, open the saved query `hcai-dev/trace-request` and paste the ID. It shows every log line and metric for that request, across guardrails, retrieval, the LLM call and agent steps.
- **Metrics come from logs:** the app writes CloudWatch Embedded Metric Format (EMF) lines, which CloudWatch turns into metrics with no agent or SDK. Question text is never logged; patient names and MRNs are redacted.

## Common operations

| Task | Command |
|---|---|
| Re-index after new guideline documents | Deploy (the pipeline runs ingestion), or `curl -X POST $API_URL/ingest -H "X-Admin-Token: <token>"` |
| Roll back | Re-run the Deploy workflow on an older commit (images are immutable), or `aws ecs update-service ... --task-definition <family>:<older revision>` |
| Rotate the Claude API key | Create a new key in the Anthropic console → `put-secret-value` → `--force-new-deployment` → revoke the old key |
| Scale | `desired_count` / `max_count` in tfvars → `terraform apply` |
| Change embedding model | Change `EMBEDDING_PROVIDER` → deploy; the ingestion task rebuilds the pgvector table (dimension change is detected); re-run `make calibrate` and update `ABSTAIN_MIN_SCORE` |
| Paused agent reviews | Stored in Postgres (`checkpoints*` tables), so they survive deploys and restarts and any task can resume any thread |

## Incident checklist

1. **Alarm fires:** open the dashboard and find the time window.
2. **Errors:** run Logs Insights `filter level = "ERROR"`, pick a `request_id`, then use the `trace-request` query.
3. **Latency:** check `RetrievalMs` versus total latency. If retrieval is fine, it's the LLM provider (check status.anthropic.com or the Bedrock quotas).
4. **Low groundedness spike:** usually a bad ingestion (wrong or empty documents) or a model change. Roll back, then run `make eval` locally against the same data.
5. **Suspected data exposure:** follow the breach procedure (DPO immediately; supervisory authority within 72 hours if required).

## Tear down

```bash
terraform -chdir=infra/terraform destroy -var-file=environments/dev.tfvars
```

With `db_deletion_protection = false` (the default), this removes everything, including the database (no final snapshot) and ECR images. Secrets and the KMS key enter their 7-day recovery and deletion windows.

## Before real data

HTTPS (`certificate_arn`), AWS WAF on the ALB, tasks in private subnets behind NAT or VPC endpoints, `multi_az = true`, `db_deletion_protection = true`, ALB access logs, a BAA/DPA-covered account, a DPIA, the penetration-testing process, clinical safety sign-off (e.g. DCB0129/0160-style hazard log) and real user access control (the demo API has no end-user authentication).
