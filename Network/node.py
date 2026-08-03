from dataclasses import dataclass


@dataclass
class Node:
    """
    Lightning Network Node Model

    Represents a node in the LN topology.
    """

    node_id: str

    country: str = "US"

    latitude: float = 0.0

    longitude: float = 0.0

    carbon_intensity: float = 300.0

    online: bool = True




    # Node Status Management


    def is_available(self) -> bool:
        """
        Check whether node can participate
        in routing.
        """

        return self.online



    def go_offline(self):
        """
        Disable node.
        """

        self.online = False



    def go_online(self):
        """
        Enable node.
        """

        self.online = True




    # Feature Extraction


    def get_features(self):
        """
        Return node features.

        Useful for RL state representation.
        """

        return {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "carbon_intensity": self.carbon_intensity,
            "online": int(self.online)
        }



    def __repr__(self):

        return (
            f"Node("
            f"id={self.node_id}, "
            f"country={self.country}, "
            f"online={self.online})"
        )