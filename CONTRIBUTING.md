# How to Contribute

This repository is currently not accepting contributions. We appreciate your
interest, but we are not accepting merge requests or issues from outside the
project team at this time.

For internal team members: use the standard GitLab merge-request workflow.
Read [`AGENTS.md`](AGENTS.md) first for setup, code conventions, and the
required pre-submit checks (`make lint-check`, `make mypy`, `make test`), and
see [Adding A New Task Or Service](docs/developer/adding-a-task-or-service.md)
if your change extends the repo with a new package.

## Code Reviews (Not Currently Accepting Public Contributions)

All submissions, including submissions by project members, require review. We
use GitLab merge requests for this purpose.

## Signing Your Work (Not Currently Accepting Public Contributions)

- We require that all contributors "sign off" on their commits. This
  certifies that the contribution is your original work, or you have rights
  to submit it under the same license, or a compatible license.

  - Any contribution which contains commits that are not signed off will not
    be accepted.

- To sign off on a commit, use the `--signoff` (or `-s`) option when
  committing your changes:

  ```bash
  git commit -s -m "Add cool feature."
  ```

  This appends the following to your commit message:

  ```text
  Signed-off-by: Your Name <your@email.com>
  ```

- Full text of the DCO:

  ```text
    Developer Certificate of Origin
    Version 1.1

    Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
    1 Letterman Drive
    Suite D4700
    San Francisco, CA, 94129

    Everyone is permitted to copy and distribute verbatim copies of this
    license document, but changing it is not allowed.
  ```

  ```text
    Developer's Certificate of Origin 1.1

    By making a contribution to this project, I certify that:

    (a) The contribution was created in whole or in part by me and I have the
        right to submit it under the open source license indicated in the file;
        or

    (b) The contribution is based upon previous work that, to the best of my
        knowledge, is covered under an appropriate open source license and I
        have the right under that license to submit that work with
        modifications, whether created in whole or in part by me, under the
        same open source license (unless I am permitted to submit under a
        different license), as indicated in the file; or

    (c) The contribution was provided directly to me by some other person who
        certified (a), (b) or (c) and I have not modified it.

    (d) I understand and agree that this project and the contribution are
        public and that a record of the contribution (including all personal
        information I submit with it, including my sign-off) is maintained
        indefinitely and may be redistributed consistent with this project or
        the open source license(s) involved.
  ```

## License

This project is licensed under the [Apache License 2.0](LICENSE). All
contributions must be licensed under Apache-2.0.

## Reporting Security Issues

Do not file public issues for security vulnerabilities. See
[`SECURITY.md`](SECURITY.md) for the private reporting path.
