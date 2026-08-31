# Security Policy

## Reporting a Vulnerability

Do not report suspected vulnerabilities in a GitLab issue, merge request,
discussion, or other public project channel.

Report potential NVIDIA product vulnerabilities through NVIDIA PSIRT:

- [Security Vulnerability Submission Form](https://www.nvidia.com/object/submit-security-vulnerability.html)
- Email: [psirt@nvidia.com](mailto:psirt@nvidia.com)
- [NVIDIA public PGP key](https://www.nvidia.com/en-us/security/pgp-key)

Include the affected branch or version, vulnerability type, reproduction steps,
potential impact, and a minimal proof of concept when available. Remove
credentials, customer data, personal data, and unrelated proprietary
information from the report.

For a security concern that is clearly limited to internal repository access or
configuration, contact the project maintainers through an approved private
NVIDIA channel.

## Coordinated Disclosure

Allow the security and project teams time to investigate and remediate before
sharing details beyond the approved response team. Follow
[NVIDIA PSIRT policies](https://www.nvidia.com/en-us/security/psirt-policies/)
for disclosure and acknowledgement practices.

## Deployment Responsibilities

Before deployment:

- Scan Python dependencies, container images, operating-system packages, and
  model artifacts with approved tools.
- Verify the source, integrity, and license of model checkpoints and datasets.
- Keep API keys and storage credentials in approved secret-management systems.
- Mount model and input data read-only whenever possible.
- Run containers with least privilege and restrict network access to required
  endpoints.
- Review generated logs and evidence for sensitive media, prompts, answers, or
  endpoint metadata.

The absence of a known vulnerability does not make a model, container, or
third-party package safe for every environment. Apply NVIDIA security
requirements and the controls required by the target deployment.
