# Complex geometry domain에서의 Elliptic PDE를 계산하기 위한 GreenNet & CouplingNet의 개발

너는 OpenAI의 codex이고 GPT5를 기반으로 하고 있어. 이 repo에서의 너의 목적은 Elliptic PDE를 계산하기 위한 operator-learning framework 개발을 돕는 것이야.

## 반드시 지켜야 할 지침

- `docs/memory.md`파일을 작성하고 매 작업마다 네가 기억해야 할 것을 정리해서 업데이트 해야 해.
- 모든 작업을 시작하기 전에 너는 항상 `docs/memory.md` 파일의 내용을 확인해야 해.
- 매 작업마다 `README.md`파일을 업데이트 해야 해.


## Python Coding을 위한 지침

- 아래의 예제를 바탕으로 로그 출력을 해줘.

  ```python
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

- 작업 디렉토리에는 터미널에 출력되는 것과 같은 로그가 파일로 저장되어야 해.

- `class`를 사용하는 것을 우선 순위에 두어야 해. 나는 OOP(Object-Oriented Programing) 방식을 선호해.

- 네가 `class`를 사용할 때에는 반드시 `Mixin` 구조를 고려해야 해.

- 같은 동작을 행하는 구조를 위해서 `dataclasses`와 `Mixin`구조를 항상 염두해두어야 해.

- 그래프를 그릴 때, 나는 `matplotlib`보다 `plotly`를 선호해.

- Python을 실행할 때는 항상  `~/.conda/envs/green_net/bin/python`의 conda 환경을 실행해.

- 매 작업 후에 린트 검사 및 타입 검사를 수행해줘. `ruff check src`, `ruff format src`, `mypy src` 명령어를 사용해줘.

- PEP8을 준수하여 4칸 들여쓰기, `PascalCase` 클래스, `snake_case` 함수 및 `UPPER_SNAKE_CASE` 상수를 사용해.

## Project 구조

- `src/`에는 core code가 저장된다.
- `cli/`에는 CLI들이 저장된다.
- `configs/`dpsms JSON 형식의 설정파일이 저장된다.
- `checkpoints/`에는 모델의 가중치, 로그 파일, 설정 파일등의 산출물이 저장된다.

## 테스트 가이드라인

- 테스트 파일은 `test/` 디렉터리에 `test_<module>.py`라는 이름으로 저장해줘.