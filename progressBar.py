from time import perf_counter

class ProgressBar:
    def __init__(self, name: str, all_cycles: int, start_time: float, bar_length: int = 40) -> None:
        self.step = 1 / all_cycles
        self.percent = 0
        self.bar_length = bar_length
        self.all_cycles = all_cycles
        self.start_time = start_time
        self.cycles_width = max(2*len(str(all_cycles)) + 3, 8)
        self.runtime_width = max(2*len(str(round((perf_counter() - start_time) * all_cycles, 1))) + 4, 9)
        print('', name.center(self.bar_length), 
              " Cycles ".center(self.cycles_width), 
              " Runtime ".center(self.runtime_width), 
              " Sample cost ", 
              " Validation cost ", 
              '',
              sep='|')
    
    def __call__(self, finished_cycles: int, sample_cost: float, val_cost: float = 1) -> None:
        runtime = perf_counter() - self.start_time
        self.percent += self.step
        cycles_line = f"{finished_cycles}/{self.all_cycles}".center(self.cycles_width)
        runtime_line = f"{round(runtime, 1)}/{round(runtime / self.percent, 1)}s".center(self.runtime_width)
        sample_cost_line = f"{round(sample_cost, 3)}".center(13)
        val_cost_line = f"{round(val_cost, 3)}".center(17)
        print(f"\r{self.bar()}{cycles_line}|{runtime_line}|{sample_cost_line}|{val_cost_line}|", flush=True, end='')
        if finished_cycles == self.all_cycles:
            print()
    
    def bar(self) -> str:
        filled = round(self.percent * self.bar_length)
        return f"|{'█'*(filled)}{' '*(self.bar_length - filled)}|"