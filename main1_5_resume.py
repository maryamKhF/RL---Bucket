"""Test the trained main1/5 model from saved artifacts; never train a policy."""

from main1_5 import main as _main


def main(argv=None):
    return _main(argv, phase="evaluate")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
