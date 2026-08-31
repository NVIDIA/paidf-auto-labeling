# Auto-Labeling Documentation

This folder is split by audience. Use both when you need to run the product
and change it.

## If you want to run auto-labeling

Start with the [User Guide](user-guide/README.md). That path covers install,
the first cookbook run, endpoints, checkpoints, remote storage, outputs, and
runtime troubleshooting.

Recommended order:

1. [Installation](user-guide/installation.md)
2. [Getting Started](user-guide/getting-started.md)
3. [Samples and Cookbooks](user-guide/samples-and-cookbooks.md)

## If you want to change Auto-Labeling

Start with [Developer Docs](developer/README.md). That path covers contributor
setup, architecture, artifact contracts, and adding a task or service.

Recommended order:

1. [Local Development](developer/local-development.md)
2. [Adding A New Task Or Service](developer/adding-a-task-or-service.md)
3. [Services Overview](developer/architecture/services-overview.md)

## Canonical Folders

- [User Guide](user-guide/README.md) — product and operator documentation
- [Developer Docs](developer/README.md) — contributor documentation, including
  a [package and service README index](developer/README.md#package-and-service-index)
- [Cookbooks](../cookbooks/README.md) — scenario configs and runbooks

## Project Policies

- [Project README](../README.md)
- [Contributing](../CONTRIBUTING.md)
- [Security Policy](../SECURITY.md)

Never store endpoint credentials in Markdown, cookbooks, payloads, or logs.
