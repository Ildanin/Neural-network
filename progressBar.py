from time import perf_counter

class ProgressBar:
    def __init__(self, name: str, total_cycles: int, validation: bool = False, bar_length: int = 40) -> None:
        self.name = name
        self.total_cycles = total_cycles
        self.validation = validation
        self.bar_length = bar_length
        self.cycle = 0
        self.start_time = perf_counter()
    
    def __call__(self, train_loss: float, test_loss: float = 1) -> None:
        runtime = perf_counter() - self.start_time
        if self.cycle == 0:
            self.update_spacing(runtime)
            self.print_init()
        self.cycle += 1
        self.percent = self.cycle / self.total_cycles
        cycles_line = f"{self.cycle}/{self.total_cycles}".center(self.cycles_width)
        runtime_line = f"{round(runtime, 1)}/{round(runtime / self.percent, 1)}s".center(self.runtime_width)
        train_loss_line = f"{round(train_loss, 3)}".center(12)
        if self.validation:
            test_loss_line = f"{round(test_loss, 3)}".center(11)
            print(f"\r{self.bar()}{cycles_line}|{runtime_line}|{train_loss_line}|{test_loss_line}|", flush=True, end='')
        else:
            print(f"\r{self.bar()}{cycles_line}|{runtime_line}|{train_loss_line}|", flush=True, end='')
        if self.cycle == self.total_cycles:
            print()
    
    def bar(self) -> str:
        filled = round(self.percent * self.bar_length)
        return f"|{'█'*(filled)}{' '*(self.bar_length - filled)}|"
    
    def update_spacing(self, runtime: float) -> None:
        self.cycles_width = max(2*len(str(self.total_cycles)) + 3, 8)
        self.runtime_width = max(2*len(str(round((runtime - self.start_time) * self.total_cycles, 1))) + 4, 9)
    
    def print_init(self) -> None:
        if self.validation:
            print('', self.name.center(self.bar_length), 
                " Cycles ".center(self.cycles_width), 
                " Runtime ".center(self.runtime_width), 
                " Train loss ", 
                " Test loss ", 
                '', sep='|')
        else:
            print('', self.name.center(self.bar_length), 
                " Cycles ".center(self.cycles_width), 
                " Runtime ".center(self.runtime_width), 
                " Train loss ", 
                '', sep='|')