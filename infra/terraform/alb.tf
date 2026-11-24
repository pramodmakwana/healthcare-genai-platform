resource "aws_lb" "api" {
  #checkov:skip=CKV_AWS_150:Controlled by var.db_deletion_protection; off for throwaway sandboxes.
  #checkov:skip=CKV_AWS_91:Access logs need an S3 bucket + policy; app-level request logs with request IDs cover the demo. Enable for production.
  #checkov:skip=CKV2_AWS_28:Ingress is IP-allowlisted; add AWS WAF (managed rule groups) before opening to the internet.
  #checkov:skip=CKV2_AWS_20:HTTP redirects to HTTPS whenever certificate_arn is set (see aws_lb_listener.http).
  name                       = local.name
  load_balancer_type         = "application"
  subnets                    = aws_subnet.public[*].id
  security_groups            = [aws_security_group.alb.id]
  drop_invalid_header_fields = true
  enable_deletion_protection = var.db_deletion_protection
  idle_timeout               = 120 # LLM calls can take tens of seconds
}

resource "aws_lb_target_group" "api" {
  #checkov:skip=CKV_AWS_378:TLS terminates at the ALB; ALB-to-task traffic stays inside the VPC.
  name                 = local.name
  port                 = 8000
  protocol             = "HTTP"
  target_type          = "ip"
  vpc_id               = aws_vpc.main.id
  deregistration_delay = 30

  health_check {
    path                = "/health"
    matcher             = "200"
    interval            = 15
    healthy_threshold   = 2
    unhealthy_threshold = 3
  }
}

# HTTP: forward when there is no certificate (demo), otherwise redirect to HTTPS.
resource "aws_lb_listener" "http" {
  #checkov:skip=CKV_AWS_2:Redirects to HTTPS when certificate_arn is set; plain HTTP only for an IP-allowlisted demo with synthetic data.
  #checkov:skip=CKV_AWS_103:Same as above; the HTTPS listener uses a TLS 1.3/1.2 policy.
  load_balancer_arn = aws_lb.api.arn
  port              = 80
  protocol          = "HTTP"

  dynamic "default_action" {
    for_each = var.certificate_arn == "" ? [1] : []
    content {
      type             = "forward"
      target_group_arn = aws_lb_target_group.api.arn
    }
  }

  dynamic "default_action" {
    for_each = var.certificate_arn == "" ? [] : [1]
    content {
      type = "redirect"
      redirect {
        port        = "443"
        protocol    = "HTTPS"
        status_code = "HTTP_301"
      }
    }
  }
}

resource "aws_lb_listener" "https" {
  count             = var.certificate_arn == "" ? 0 : 1
  load_balancer_arn = aws_lb.api.arn
  port              = 443
  protocol          = "HTTPS"
  ssl_policy        = "ELBSecurityPolicy-TLS13-1-2-2021-06"
  certificate_arn   = var.certificate_arn

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.api.arn
  }
}
