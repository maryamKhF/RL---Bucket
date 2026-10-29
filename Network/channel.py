from dataclasses import dataclass


@dataclass
class Channel:
    """
    Lightning Network Channel Model

    Represents a directed channel state
    inside the simulator.
    """

    channel_id: str

    capacity: float

    fee_base: float

    fee_rate: float

    delay: float

    failure_probability: float = 0.01

    available: bool = True

    balance_uv: float = 0.0

    balance_vu: float = 0.0

    failure_count: int = 0

    success_count: int = 0



    # Fee Calculation


    def calculate_fee(self, amount: float) -> float:
        """
        Calculate forwarding fee.
        """

        return (
            self.fee_base +
            amount * self.fee_rate
        )


    # Liquidity Checking


    def can_forward_uv(
            self,
            amount: float
    ) -> bool:
        """
        Check u -> v liquidity.
        """

        return (
            self.available and
            self.balance_uv >= amount
        )


    def can_forward_vu(
            self,
            amount: float
    ) -> bool:
        """
        Check v -> u liquidity.
        """

        return (
            self.available and
            self.balance_vu >= amount
        )



    # Payment Forwarding


    def forward_uv(
            self,
            amount: float
    ) -> bool:
        """
        Execute u -> v transfer.
        """

        if not self.can_forward_uv(amount):

            self.failure_count += 1

            return False


        self.balance_uv -= amount
        self.balance_vu += amount

        self.success_count += 1

        return True



    def forward_vu(
            self,
            amount: float
    ) -> bool:
        """
        Execute v -> u transfer.
        """

        if not self.can_forward_vu(amount):

            self.failure_count += 1

            return False


        self.balance_vu -= amount
        self.balance_uv += amount

        self.success_count += 1

        return True




    # Statistics


    def success_rate(self) -> float:
        """
        Channel reliability metric.

        Useful as RL feature.
        """

        total = (
            self.success_count +
            self.failure_count
        )

        if total == 0:
            return 0.0

        return (
            self.success_count /
            total
        )



    def liquidity_ratio(self):
        """
        Current available liquidity ratio.
        """

        return (
            self.balance_uv /
            self.capacity
            if self.capacity > 0
            else 0
        )



    # State


    def disable(self):

        self.available = False



    def enable(self):

        self.available = True



    def reset_statistics(self):

        self.failure_count = 0

        self.success_count = 0