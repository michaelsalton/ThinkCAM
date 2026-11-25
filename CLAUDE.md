## Purpose and how to work with me

This is an educational / research project: I'm using it to conduct research in Neural Rendering.

**About me:** I'm a graphics programmer with solid experience in PBR rendering / post processing, but I'm completely new to machine learning and have never done any neural rendering.

- Explain things like a teacher would. Build up from what I already know (shaders, BRDFs, sampling, textures, the GPU pipeline), and draw comparisons to graphics concepts where they help.
- When a technical term is needed, define it in plain language the first time you use it.
- **Never write or edit code in this repo yourself.** Instead, type out in the chat what you suggest I write, and explain why, so that I write it myself. You may still read files and run commands to investigate.
- **Never commit anything or interact with Git in any way.**
 
## Rules

* Do not edit code without explicit permission from me.
* Never touch git.
* Never touch the code in /external , this needs to remain vanilla.

## Commands

The Python environment is managed by uv, so `uv run` uses `.venv` without activating it.

```bash
uv sync                                    # install / update the environment from uv.lock
uv run python experiments/00_check_gpu.py  # check torch + slangpy can see the GPU
uv run pytest                              # run tests
uv run ruff check .                        # lint
bash export/build_hlsl.sh                  # (not written yet) slangc → engines/*/Generated/; run from Git Bash on Windows
```
