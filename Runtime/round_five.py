"""Final continual stage, using the optimized fourth-stage execution path."""

from Runtime.round_four import MainFourOfFive


class MainFiveOfFive(MainFourOfFive):
    round_number = 5
    previous_round_number = 4
    previous_runner_class = MainFourOfFive
