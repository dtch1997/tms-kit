# Setup

## Install PDM 

Our repository uses [PDM](https://pdm-project.org/en/latest/) to manage dependencies. 

This has several benefits. 
- **Package compatibility**. When adding a new package, PDM resolves package versions to ensure all packages are mutually compatible. 
- **Reproducibility**. PDM pins packages to specific versions, ensuring reproducibility.  
- **Cross-platform Compatibility**. PDM pins versions for different platforms simultaneously (e.g. Linux vs MacOS vs Windows), ensuring cross-platformness. 

To install PDM, follow [official instructions](https://pdm-project.org/en/latest/#installation). 

To ensure PDM is installed correctly, you can run a test command
```
pdm --version
# PDM, version 2.18.0
```

## Dev Environment

Under the hood, PDM re-uses `venv` to store the Python interpreter and all dependencies. 

You can use this `venv` in the usual way: 
```bash
source .venv/bin/activate
```

Dependencies will be pinned to specific versions in `pdm.lock`, making setup reproducible across different machines and platforms. 

Basic usage:
```bash
pdm install # Install dependencies from pdm.lock 
pdm add <PACKAGE> # Add a new package and update pdm.lock
```

## Barebones VirtualEnv

Some environments (e.g. the cluster) do not come with PDM. Our project can also be installed as a local virtualenv. 

```bash
# Assume you have created your venv locally
source .venv/bin/activate
pip install -e .
```