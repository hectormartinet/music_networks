import time


class Timer:

    def __init__(self, name):
        self.name = name
        self.start_time = None
        self.run_time = 0

    def start(self):
        if self.start_time is not None:
            raise Exception("The timer has already started")
        self.start_time = time.time()

    def end(self):
        if self.start_time is None:
            raise Exception("The timer has not started")
        self.run_time += time.time()-self.start_time
        self.start_time = None
    
    def get_run_time(self):
        return self.run_time

class MultiTimer:
    
    def __init__(self):
        self.timers = {}

    def create(self, name):
        self.timers[name] = Timer(name)

    def start(self, name):
        if not name in self.timers.keys():
            self.create(name)
        self.timers[name].start()
    
    def end(self, name):
        if not name in self.timers.keys():
            raise Exception(f"Timer {name} doesn't exist")
        self.timers[name].end()
    
    def get_run_time(self, name):
        if not name in self.timers.keys():
            raise Exception(f"Timer {name} doesn't exist")
        return self.timers[name].get_run_time()
    
    def print_times(self):
        for name,timer in self.timers.items():
            print(f"{name}:{timer.get_run_time()}")