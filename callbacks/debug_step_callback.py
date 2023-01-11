from typing import Any


class Callback():
    pass
class JustTestCanRun(Callback):
    def __init__(self, debug=True) -> None:
        super().__init__()
        self.max_step = 50 if debug else float('inf')  
    def step(self, step, **kargs):
        if step > self.max_step:
            raise StopIteration
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.step(*args, **kwds)
    