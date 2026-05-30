<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Contributing

Thank you for your interest in contributing to `paidf-auto-labeling`.

## License

This project is licensed under the [Apache License 2.0](LICENSE). **All contributions must be licensed under Apache-2.0.** By submitting a contribution, you agree that your contribution will be made available under the terms of the Apache License 2.0 and that you have the right to do so.

Contributions that include third-party source code under a different license can only be accepted if (a) the third-party license is compatible with Apache-2.0 (e.g. MIT, BSD), (b) the upstream license text and copyright notices are preserved in the contributed files and in `THIRD_PARTY_NOTICES.md`, and (c) the SPDX-License-Identifier of each affected file uses an expression that includes both licenses (for example `MIT AND Apache-2.0`).

## Developer Certificate of Origin (DCO)

This project uses the [Developer Certificate of Origin](https://developercertificate.org/) (DCO) instead of a Contributor License Agreement. By making a contribution to this project, you certify the following:

> Developer Certificate of Origin
> Version 1.1
>
> Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
>
> Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
>
>
> Developer's Certificate of Origin 1.1
>
> By making a contribution to this project, I certify that:
>
> (a) The contribution was created in whole or in part by me and I have the right to submit it under the open source license indicated in the file; or
>
> (b) The contribution is based upon previous work that, to the best of my knowledge, is covered under an appropriate open source license and I have the right under that license to submit that work with modifications, whether created in whole or in part by me, under the same open source license (unless I am permitted to submit under a different license), as indicated in the file; or
>
> (c) The contribution was provided directly to me by some other person who certified (a), (b) or (c) and I have not modified it.
>
> (d) I understand and agree that this project and the contribution are public and that a record of the contribution (including all personal information I submit with it, including my sign-off) is maintained indefinitely and may be redistributed consistent with this project or the open source license(s) involved.

### Signing off your commits

Every commit must include a `Signed-off-by` trailer that matches the author's name and a verifiable email address:

```bash
git commit -s -m "your commit message"
```

This adds a line of the form:

```text
Signed-off-by: Your Name <your.email@example.com>
```

at the end of the commit message. Commits without a valid sign-off will be rejected.

## License headers

Every source file authored or modified by contributors must carry an SPDX header at the top of the file using the form below (adapt the comment syntax to the file type):

```text
SPDX-FileCopyrightText: Copyright (c) <year> <author or organization>. All rights reserved.
SPDX-License-Identifier: Apache-2.0
```

Files vendored from third-party projects must retain the upstream copyright and license text. If you make non-trivial modifications to such a file, use a dual SPDX expression (for example `MIT AND Apache-2.0`) and add an NVIDIA copyright line alongside the upstream one.

### Exception: VLM/LLM prompt templates

Markdown files under `cookbooks/**/prompts/**/*.md` are loaded verbatim as prompts for vision-language / language models. Adding an SPDX comment to those files would inject license boilerplate into the model input and degrade prompt quality. These prompt files therefore intentionally **omit** SPDX headers; their license (Apache-2.0) is governed by the root `LICENSE` file and the entry for `cookbooks/` in `THIRD_PARTY_NOTICES.md`.

## How to file a change

1. Fork the repository and create a topic branch.
2. Make your changes; keep diffs focused and add tests where relevant.
3. Run `ruff check` and `pytest` locally before opening a merge request.
4. Update `README.md`, `docs/`, and `THIRD_PARTY_NOTICES.md` if your change affects user-facing behavior, dependencies, or vendored third-party material.
5. Ensure every commit is signed off (see above).
6. Open a merge request describing the motivation, the change, and how it was tested.

## Reporting security issues

Please do not file public issues for security vulnerabilities. Email the maintainers directly with the details and we will coordinate disclosure.
