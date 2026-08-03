# RL/environment.py

import gymnasium as gym
from gymnasium import spaces

import numpy as np


from Pathfinding.dijkstra import Dijkstra
from Pathfinding.top_k_paths import top_k_paths

from Simulation.payment_simulator import simulate_payment

from .state import State
from .reward import calculate_reward



class RoutingEnv(gym.Env):

    metadata = {
        "render_modes": []
    }



    def __init__(
        self,
        G,
        transactions,
        heuristic_fn,
        config,
        mode="hybrid",
        bucket=None
    ):

        super().__init__()


        self.G = G

        self.transactions = transactions

        self.heuristic = heuristic_fn

        self.cfg = config

        self.mode = mode

        self.bucket = bucket


        # Dijkstra Router

        self.router = Dijkstra(G)



        self.tx_index = 0

        self.current_tx = None


        self.current_paths = []



        # ------------------------------------------------
        # Observation Space
        # ------------------------------------------------

        obs_dim = (

            config["graph"]["neighborhood_k"]

            *

            config["graph"]["neighborhood_m"]

            *

            8

        )



        self.observation_space = spaces.Box(

            low=-np.inf,

            high=np.inf,

            shape=(obs_dim,),

            dtype=np.float32

        )



        # ------------------------------------------------
        # Action Space
        #
        # PPO learns eta parameter
        # ------------------------------------------------

        self.action_space = spaces.Box(

            low=np.array(
                [0.0],
                dtype=np.float32
            ),

            high=np.array(
                [1.0],
                dtype=np.float32
            ),

            dtype=np.float32

        )




    # =====================================================
    # RESET
    # =====================================================

    def reset(
        self,
        seed=None,
        options=None
    ):


        super().reset(seed=seed)


        self.tx_index = 0


        self.current_tx = (

            self.transactions[
                self.tx_index
            ]

        )


        self.current_paths = []



        observation = self._build_state()



        return observation, {}





    # =====================================================
    # STEP
    # =====================================================

    def step(
        self,
        action
    ):


        tx = self.current_tx



        # PPO action

        eta = float(

            np.asarray(action)
            .reshape(-1)[0]

        )



        # ------------------------------------------------
        # Candidate paths
        # ------------------------------------------------

        try:

            self.current_paths = top_k_paths(

                self.G,

                tx.source,

                tx.destination,

                tx.amount,

                self.cfg["graph"]["top_k"]

            )


        except Exception:


            self.current_paths = []





        # ------------------------------------------------
        # Dijkstra Routing
        # ------------------------------------------------

        route = self.router.shortest_path(

            tx.source,

            tx.destination,

            tx.amount,

            self.heuristic,

            eta,

            self.cfg["graph"]["max_hops"]

        )




        if route["success"]:


            path = route["path"]

            edges = route["edges"]



            result = simulate_payment(

                self.G,

                path,

                edges,

                tx.amount

            )



            reward = calculate_reward(

                success=result.success,

                path_length=len(edges),

                carbon_intensity=result.carbon,

                scale=self.cfg["simulation"]
                ["base_reward_scale"]

            )




            info = {


                "success":
                    True,


                "eta":
                    eta,


                "path":
                    path,


                "path_length":
                    len(edges),


                "fee":
                    route["total_fee"],


                "delay":
                    route["total_delay"],


                "carbon":
                    result.carbon

            }




        else:


            reward = calculate_reward(

                success=False,

                path_length=0,

                carbon_intensity=0

            )


            info = {


                "success":
                    False,


                "eta":
                    eta,


                "path":
                    None,


                "path_length":
                    0,


                "fee":
                    0,


                "delay":
                    0,


                "carbon":
                    0

            }




        
        # Next Transaction
       

        self.tx_index += 1



        terminated = (

            self.tx_index

            >=

            len(self.transactions)

        )



        truncated = False




        if terminated:


            observation = np.zeros(

                self.observation_space.shape,

                dtype=np.float32

            )


        else:


            self.current_tx = (

                self.transactions[
                    self.tx_index
                ]

            )


            observation = self._build_state()




        return (

            observation,

            reward,

            terminated,

            truncated,

            info

        )





    
    # STATE BUILDER
   

    def _build_state(self):

      tx = self.current_tx


      state = State(

        self.G,

        tx.source,

        tx.destination,


        transaction={

            "source": tx.source,

            "destination": tx.destination,

            "amount": tx.amount

        },


        candidate_paths=self.current_paths,


        simulation_info={

            "failure_probability":0,

            "average_delay":0

        },


        bucket_info={

            "bucket_size":
                self.bucket.size()
                if self.bucket
                else 0,


            "backtrack_count":0

        },


        k=self.cfg["graph"]["neighborhood_k"],


        m=self.cfg["graph"]["neighborhood_m"],


        radius=self.cfg["graph"]["ego_radius"]

    )


      return state.vector()