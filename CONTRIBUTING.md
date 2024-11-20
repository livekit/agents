# Contributing to LiveKit

LiveKit is an open-source platform for building real-time audio/video experiences. The Agents framework is under active development in a rapidly evolving field. We welcome and appreciate contributions of any kind, be it feedback, bugfixes, features, new plugins and tools, or better documentation.

## 🤝 What We're Looking For

- Bug fixes and improvements to existing features
- New features and plugins
- Documentation improvements and examples
- Test coverage improvements
- Performance optimizations

## 👩‍💻 How to contribute

Please follow the [fork and pull request](https://docs.github.com/en/get-started/quickstart/contributing-to-projects) workflow:

1. Fork the repository
2. Create a new branch for your feature: `git checkout -b feature-name`
3. Add your feature or improvement
4. Commit your changes: `git commit -m 'Add new feature'`
5. Push to your fork: `git push origin feature-name`
6. Open a pull request with a detailed description of changes

## 🔧 Development setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/livekit.git
   cd livekit
   ```

2. Create a virtual environment:
   - For Unix: `./scripts/create_venv.sh`
   - For Windows: `.\scripts\create_venv.bat`
   - This setup will:
     - Create a `livekitenv` virtual environment
     - Install required packages
     - Install the `livekit/agents` package in editable mode

3. Activate the virtual environment:
   - Unix: `source livekitenv/bin/activate`
   - Windows: `livekitenv\Scripts\activate`


## ✅ Testing

1. Run the test suite:
   ```bash
   pytest
   ```

## 🎨 Code Style and Validation

Before submitting a pull request, ensure your code meets our quality standards:

1. Run the formatting script:
   - Unix: `./scripts/format.sh`
   - Windows: `.\scripts\format.bat`

These scripts will:
- Format code with `ruff`
- Run static type checks with `mypy`
- Execute unit tests with `pytest`

## 🐛 Troubleshooting

Common issues and solutions:
- Virtual environment creation fails: Ensure Python 3.8+ is installed
- Import errors: Check if virtual environment is activated
- Test failures: Ensure all dependencies are installed

## 📚 Resources

- [Documentation](https://docs.livekit.io/agents)
- [Python SDK](https://docs.livekit.io/python/livekit/)
- [Slack Community](https://livekit.io/join-slack)

## ❓ Getting Help

- Check existing issues and documentation first
- Open an issue for bugs or feature requests
- Join our [Slack community](https://livekit.io/join-slack) for questions

