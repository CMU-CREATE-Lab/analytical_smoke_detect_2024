import pytest
import yappi
import os

def pytest_addoption(parser):
    print("Adding yappi option")
    parser.addoption(
        "--yappi",
        action="store_true",
        default=False,
        help="enable yappi CPU profiling"
    )

def pytest_runtest_setup(item):
    if item.config.getoption("--yappi"):
        print(f"Starting yappi profiler for {item.name}")
        yappi.set_clock_type("cpu")
        yappi.start()

def pytest_runtest_teardown(item):
    if item.config.getoption("--yappi"):
        print(f"Test completed, stopping yappi for {item.name}")
        yappi.stop()
        
        stats = yappi.get_func_stats()
        stats.print_all()
        
        output_file = f"yappi_{item.name}.prof"
        stats.save(output_file, type="pstat")
        print(f"Saved profile to: {output_file}")
        
        yappi.clear_stats()
