variable "project" {
  type    = string
  default = "hcai"
}

variable "environment" {
  type    = string
  default = "dev"
}

variable "region" {
  description = "eu-west-1 (Ireland) keeps data in the EU"
  type        = string
  default     = "eu-west-1"
}

variable "availability_zones" {
  description = "Two AZs to use. Empty = first two available in the region."
  type        = list(string)
  default     = []
}

variable "vpc_cidr" {
  type    = string
  default = "10.20.0.0/16"
}

variable "allowed_ingress_cidrs" {
  description = "Who can reach the load balancer. Restrict to your IP (x.x.x.x/32) for a demo."
  type        = list(string)
}

variable "certificate_arn" {
  description = "ACM certificate for HTTPS. Empty = HTTP only (demo with synthetic data only)."
  type        = string
  default     = ""
}

variable "image_tag" {
  description = "Initial image tag; later deploys come from GitHub Actions."
  type        = string
  default     = "bootstrap"
}

variable "task_cpu" {
  type    = number
  default = 512
}

variable "task_memory" {
  type    = number
  default = 1024
}

variable "desired_count" {
  type    = number
  default = 1
}

variable "max_count" {
  type    = number
  default = 3
}

variable "db_instance_class" {
  type    = string
  default = "db.t4g.micro"
}

variable "db_deletion_protection" {
  description = "Set true for anything beyond a throwaway sandbox."
  type        = bool
  default     = false
}

variable "llm_provider" {
  description = "anthropic (Claude API, key in Secrets Manager) or bedrock (Claude on Bedrock; data stays in AWS)."
  type        = string
  default     = "anthropic"

  validation {
    condition     = contains(["anthropic", "bedrock", "mock"], var.llm_provider)
    error_message = "llm_provider must be anthropic, bedrock or mock."
  }
}

variable "llm_model" {
  description = "Claude API model name, or the Bedrock model / inference-profile ID (copy it from the Bedrock console)."
  type        = string
  default     = "claude-sonnet-5-5"
}

variable "llm_fast_model" {
  type    = string
  default = "claude-haiku-5-5"
}

variable "github_repository" {
  description = "owner/repo allowed to deploy via OIDC"
  type        = string
  default     = "pramodmakwana/healthcare-genai-platform"
}

variable "create_github_oidc_provider" {
  description = "Only one GitHub OIDC provider can exist per AWS account; set false if it already exists."
  type        = bool
  default     = true
}

variable "alarm_email" {
  description = "Email for CloudWatch alarms (confirm the subscription email). Empty = no subscription."
  type        = string
  default     = ""
}

variable "log_retention_days" {
  type    = number
  default = 365
}
