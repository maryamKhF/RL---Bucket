"""Stage three continues stage two's PPO agent, which already learned in stage one."""

from Runtime.round_two import MainTwoOfFive


class MainThreeOfFive(MainTwoOfFive):
    """Reuse the tested continual runner, including data, optimizer and rollouts."""

    round_number = 3
    previous_round_number = 2
    previous_runner_class = MainTwoOfFive

    def _configuration(self):
        cfg = super()._configuration()
        # A separate reproducible test scenario; a larger seed is not a
        # difficulty parameter and does not change the failure probabilities.
        cfg["evaluation_seed"] = int(cfg["seed"]) + 1000
        return cfg
