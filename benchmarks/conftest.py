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
        print(bench)
        params = [i.split("[")[0] for i in bench["name"].split("_")[1:]]
        values = bench["name"].split("[")[1].split("]")[0].split("-")
        fname = ":".join(params)
        with open("metrics.perf", 'a') as fp:
            fp.write("time %s %s %f\n" % (
                fname, ':'.join(values), bench["mean"]))
    yield
