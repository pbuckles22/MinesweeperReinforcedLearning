import warnings

def pytest_configure(config):
    # Suppress pkg_resources deprecation warning from pygame
    warnings.filterwarnings("ignore", category=UserWarning, message="pkg_resources is deprecated")
    warnings.filterwarnings("ignore", category=UserWarning, message=".*pkg_resources.*deprecated.*")
    warnings.filterwarnings("ignore", category=UserWarning, message=".*pkg_resources.*slated for removal.*") 