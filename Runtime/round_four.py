"""Fourth continual stage: retain the learned agent and remove rollout barriers."""

from Runtime.round_three import MainThreeOfFive


class MainFourOfFive(MainThreeOfFive):
    round_number = 4
    previous_round_number = 3
    previous_runner_class = MainThreeOfFive

    def _configuration(self):
        cfg = super()._configuration()
        cfg["rl"]["rollout_collector"] = "ready-balanced-v1"
        cfg["evaluation_graph_clone"] = "pickle-snapshot-v1"
        cfg["evaluation_journal"] = "wal-full-v1"
        return cfg
