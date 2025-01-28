# Stopwatch takes a name, or a label_fn and a boolean enable.  If enable is False, the context manager does nothing.  Otherwise, it measures the time taken to execute the code block and prints the time taken.
# label_fn is a function that takes a dataclass with fields wall_time, cpu_time, avg_cpu_used, and cpu_count, and returns a string to output

import sys
import time

import psutil

class Stopwatch:
    def __init__(self, name, print_stats=True):
        self.name = name
        self.stats_msg = None
        self.print_stats = print_stats

    def __enter__(self):
        self.start_wall_time = time.time()
        self.start_cpu_time = psutil.Process().cpu_times().user + psutil.Process().cpu_times().system
        self.start_cpu_count = psutil.cpu_count()
        return self

    def set_stats_msg(self, stats_msg):
        self.stats_msg = stats_msg
    
    def start(self):
        self.start_wall_time = time.time()
        self.start_cpu_time = psutil.Process().cpu_times().user + psutil.Process().cpu_times().system
        self.start_cpu_count = psutil.cpu_count()
        
    def wall_elapsed(self):
        return time.time() - self.start_wall_time

    def cpu_elapsed(self):
        end_cpu_time = psutil.Process().cpu_times().user + psutil.Process().cpu_times().system
        return end_cpu_time - self.start_cpu_time

    def __exit__(self, type, value, traceback):
        msg =  self.stats_msg = f'{self.name}: {self.wall_elapsed():.1f} seconds, {self.cpu_elapsed():.1f} seconds CPU' 
        if self.stats_msg is not None:
            msg += f', {self.stats_msg}'

        if self.print_stats:
            sys.stdout.write('%s: %s\n' % (self.name, self.stats_msg))
            sys.stdout.flush()
