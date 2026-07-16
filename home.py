import os


def get_source_root():
    """Return the repository root that contains train.py and configs/."""
    return os.path.dirname(os.path.realpath(__file__))


def get_project_base():
    """Return the root used for experiment outputs.

    Data location is configured independently through ``--data-root`` or
    ``LOGFA_DATA_ROOT``.
    """
    configured = os.environ.get("LOGFA_OUTPUT_ROOT")
    if configured:
        return os.path.abspath(os.path.expanduser(configured)) + os.sep
    return get_source_root() + os.sep


if __name__ == "__main__":
    print(get_project_base())
