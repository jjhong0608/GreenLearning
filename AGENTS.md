# Resaerch Mode

Mission: Exhaustive information gathering through aggressive tool usage.

## Core Protocol
Mandatory Search: 
- Search for EVERY query, even simple ones. 
- Target 100+ tool uses per complex task. 
- You always carefully review the question befere answering.
- You should use tools as much as possible, ideally more than 100 times.
- You should also implement your own test first before attempting the problem.
- You should review and evaluate your answer and answer until your confidnece and certainty reach no less than 80%

**Extended Thinking**: Enabled. Use thinking blocks to plan searches, verify consistency.

## Execution Pattern
**For Every Query**:
- Initial broad seraches.
- Targeted specific searches.
- Verification searches.
- Comparative searches across sources.
- Recent update searches.
- Alternative angle searches.

Response Format:
- Direct 1-2 sentences answer first.
- Use markdown format: headers(###), lists, tables.
- inline citations `[1][2]`.
- Bold critical terms only.

## Quality Standards

- Search before answering — Always.
- More searches=better answer.
- When sources conflict, search more.
- Prioritize accuracy over speed.
- Default to over-researching
- Never guess — search instead

Philosophy: Treat search as unlimited resource. 

## Instruction for Python coding
- Print a log using below code example:
  ```Python
  import logging
  from rich.logging import RichHandler
  
  logger = logging.getLogger(__name__)
  handler = RichHandler(
  rich_tracebacks=True,
  show_path=True,
  omit_repeated_times=False,
  )
  formatter = logging.Formatter("%(funcName)s - %(message)s")
  handler.setFormatter(formatter)
  handler.setLevel(logging.DEBUG)
  logger.addHandler(handler)
  logger.propagate = False
  logger.setLevel(logging.DEBUG)
  logging.root.handlers.clear()
  ```
- Same logs printing in terminal have to be saved in working directiry.
- You always use `class` instead of function. (I prefer Object-Oriented Programming)
- When you use `class`, you should consider MixIn structure.
- Use dataclasses and MixIn structure when sharing behaviors
- For drawing the graph, use `Plotly` library instaed of `matplotlib`.
- To excute Python, first search a virtual environment in `.venv` directory in the project directory.


# Repository Guidelines

## Goal
Develop a deep learning model for Poisson's partial differential equation,
$$
\nabla\cdot(\kappa(\mathbf{x})\nabla u(\mathbf{x}))=f(\mathbf{x})
$$
with Dirichlet boundary condition. Your model is constructed by inspired Axial Green's Function Method (AGM). This method repsents the solution as the integration of the one-dimensional Green's function with source term. For this method, the governing equation should be decomposed along axes. The paper about AGM named `Axial Green s function method for multi‐dimensional elliptic boundary value.pdf` is in `refenreces` folder. Your model is based on the paper named `DD29_proceedings_revision_v1.pdf` in `references` folder. The code already work is in `/home/jjhong0608/Documents/GreenONet`

## Project Structures
- `src/` hosts the core code.
- `cli/` hosts CLIs
- `configs/` stores JSON style configuration file for training.
- `checkpoints/` stores output file such as model's weights, log files, configuration files used for training.

## Build, Test and Development Command
- Use virtual environtment; `source .venv/bin/activate`.
- Lint and type-check before running; `ruff check src`, `ruff format src`, `mypy src`.
- After modify or add code, you should always upate README.md file.
- After modify or add code, you should always lint and type-check usinhg `ruff` and `mypy`.

## Code style & Naming Conventions
- Follow PEP8 with four-space indentation, `PascalCase` classes, `snake_case` functions, and `UPPER_SNAKE_CASE` constants.
- Use dataclasses and mixin when sharing behaviors

## Testing Guidelines
- Place test in `test/` with filenames `test_<module>.py`

## Commit & Pull Request Guidlines
- Don't use git in this project.
- No commit & No pull