import warnings

def pytest_configure(config):
    warnings.filterwarnings("ignore", category=UserWarning, message="pkg_resources is deprecated") 