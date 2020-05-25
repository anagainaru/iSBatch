import pytest
import os


@pytest.hookimpl()
def pytest_sessionstart(session):
    try:
        os.remove("metrics.perf")
    except:
        pass


@pytest.mark.hookwrapper
def pytest_benchmark_group_stats(config, benchmarks, group_by):
    for bench in benchmarks:
        fname = ":".join([i.split("[")[0]
                          for i in bench["name"].split("_")[1:]])
        with open("metrics.perf", 'a') as fp:
            fp.write("time %s %s %f\n" % (
                fname,
                ':'.join([str(bench["params"][i]) for i in bench["params"]]),
                bench["mean"]))
    yield
