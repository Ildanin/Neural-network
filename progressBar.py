from time import perf_counter

class ProgressBar:
    def __init__(self, name: str, total_cycles: int, start_time: float, bar_length: int = 40) -> None:
        self.cycle = 0
        self.bar_length = bar_length
        self.total_cycles = total_cycles
        self.start_time = start_time
        self.cycles_width = max(2*len(str(total_cycles)) + 3, 8)
        self.runtime_width = max(2*len(str(round((perf_counter() - start_time) * total_cycles, 1))) + 4, 9)
        print('', name.center(self.bar_length), 
              " Cycles ".center(self.cycles_width), 
              " Runtime ".center(self.runtime_width), 
              " Train loss ", 
              " Test loss ", 
              '',
              sep='|')
    
    def __call__(self, train_loss: float, test_loss: float = 1) -> None:
        runtime = perf_counter() - self.start_time
        self.cycle += 1
        self.percent = self.cycle / self.total_cycles
        cycles_line = f"{self.cycle}/{self.total_cycles}".center(self.cycles_width)
        runtime_line = f"{round(runtime, 1)}/{round(runtime / self.percent, 1)}s".center(self.runtime_width)
        train_loss_line = f"{round(train_loss, 3)}".center(12)
        test_loss_line = f"{round(test_loss, 3)}".center(11)
        print(f"\r{self.bar()}{cycles_line}|{runtime_line}|{train_loss_line}|{test_loss_line}|", flush=True, end='')
        if self.cycle == self.total_cycles:
            print()
    
    def bar(self) -> str:
        filled = round(self.percent * self.bar_length)
        return f"|{'█'*(filled)}{' '*(self.bar_length - filled)}|"