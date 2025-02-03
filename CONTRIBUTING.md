# Contributing to livekit/agents

The LiveKit Agents framework is an open-source project, and we welcome any contribution from anyone
willing to work in good faith with the community. No contribution is too small!

## Code of Conduct

The LiveKit Agents project has a [Code of Conduct](/CODE_OF_CONDUCT.md) to which all contributors
must adhere.

## Contribute code

There are many ways you can contribute code to the project:

- **Write a plugin**: if there is a TTS/STT/LLM provider you use that isn't on our plugins list,
  feel free to write a plugin for it! Refer to the source code of similar plugins to see how they're
  built.

- **Fix bugs**: we strive to make this framework as reliable as possible, and we'd appreciate your
  help with squashing bugs and improving stability. Follow the guidelines below for information
  about authoring pull requests.

- **Add new features**: we're open to adding new features to the framework, though we ask that you
  open an issue first to discuss the viability and scope of the new functionality before starting
  work.

Our continuous integration requires a few additional code quality steps for your pull request to
be approved:

- Run `ruff check --fix` and `ruff format` before committing your changes to ensure consistent file
  formatting and best practices.

- If writing new methods/enums/classes, document them. This project uses
  [pdoc3](https://pdoc3.github.io/pdoc/) for automatic API documentation generation, and every new
  addition has to be properly documented.

- On your first pull request, the CLA Assistant bot will give you a link to sign this project's
  Contributor License Agreement, required to add your code to the repository.

- There's no need to mess around with `CHANGELOG.md` or package manifests — we have a bot handle
  that for us. A maintainer will add the necessary notes before merging.

## Assist others in the community

If you can't contribute code, you can still help us greatly by helping out community members who
may have questions about the framework and how to use it. Join the `#agents` channel on
[our Slack](https://livekit.io/join-slack).
