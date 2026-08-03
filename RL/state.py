import numpy as np
import networkx as nx

from Network.topology import geo_features



# --------------------------------------------------
# State Configuration
# --------------------------------------------------

K = 15

M = 5


BASE_FEATURES = 8

EXTRA_FEATURES = 8

FEATURES = BASE_FEATURES + EXTRA_FEATURES



# Feature order:
#
# 0  fee
# 1  delay
# 2  availability
# 3  distance
# 4  carbon
# 5  incoming_capacity
# 6  outgoing_capacity
# 7  node_degree
#
# 8  transaction_amount
# 9  source_degree
# 10 destination_degree
# 11 candidate_path_count
# 12 failure_probability
# 13 average_delay
# 14 bucket_size
# 15 backtrack_count



FEE_SCALE = 10000

DELAY_SCALE = 10

DISTANCE_SCALE = 10000

CARBON_SCALE = 1000





# ==================================================
# State Class
# ==================================================


class State:


    def __init__(
        self,
        G,
        source,
        destination,
        transaction=None,
        candidate_paths=None,
        simulation_info=None,
        bucket_info=None,
        k=K,
        m=M,
        radius=2
    ):


        self.G = G

        self.source = source

        self.destination = destination


        self.transaction = transaction or {}

        self.candidate_paths = candidate_paths or []


        self.simulation_info = simulation_info or {}

        self.bucket_info = bucket_info or {}


        self.k = k

        self.m = m

        self.radius = radius





    # --------------------------------------------------
    # Complete State Vector
    # --------------------------------------------------


    def vector(self):


        neighborhood = self.neighborhood_state()



        global_features = np.array(

            [

                # transaction amount

                self.transaction.get(
                    "amount",
                    0
                )
                /
                100000,



                # source degree

                self.G.degree(
                    self.source
                )
                if self.G.has_node(
                    self.source
                )
                else 0,



                # destination degree

                self.G.degree(
                    self.destination
                )
                if self.G.has_node(
                    self.destination
                )
                else 0,



                # candidate paths

                len(
                    self.candidate_paths
                ),



                # failure probability

                self.simulation_info.get(
                    "failure_probability",
                    0
                ),



                # average delay

                self.simulation_info.get(
                    "average_delay",
                    0
                )
                /
                DELAY_SCALE,



                # bucket size

                self.bucket_info.get(
                    "bucket_size",
                    0
                ),



                # backtrack count

                self.bucket_info.get(
                    "backtrack_count",
                    0
                )

            ],

            dtype=np.float32

        )



        return np.concatenate(

            [

                neighborhood,

                global_features

            ]

        )





    # --------------------------------------------------
    # Neighborhood State
    # --------------------------------------------------


    def neighborhood_state(self):


        nodes = set()



        for center in [

            self.source,

            self.destination

        ]:


            if self.G.has_node(center):


                nodes.update(

                    nx_ego_nodes(

                        self.G,

                        center,

                        self.radius

                    )

                )




        scored = []



        for node in nodes:


            if node in (

                self.source,

                self.destination

            ):

                continue



            try:


                distance = nx_shortest_hops(

                    self.G,

                    self.source,

                    node

                )


            except nx.NetworkXNoPath:


                distance = 999





            scored.append(

                (

                    distance,

                    -self.G.degree(node),

                    node

                )

            )




        selected = [

            x[2]

            for x in sorted(scored)

            [

                :

                self.k*self.m

            ]

        ]





        rows = []





        for node in selected:



            edges = get_node_edges(

                self.G,

                node

            )




            for u,v,key,data in edges[:self.m]:


                try:


                    dist, carbon, ic, icont = geo_features(

                        self.G,

                        u,

                        v

                    )


                except Exception:


                    dist = 0

                    carbon = 0

                    ic = 0

                    icont = 0





                row = [

                    data.get(
                        "fee_base",
                        0
                    )
                    /
                    FEE_SCALE,



                    data.get(
                        "delay",
                        0
                    )
                    /
                    DELAY_SCALE,



                    float(

                        data.get(
                            "active",
                            1
                        )

                    ),



                    dist
                    /
                    DISTANCE_SCALE,



                    carbon
                    /
                    CARBON_SCALE,



                    ic,



                    icont,



                    self.G.degree(node)
                    /
                    max(
                        self.G.number_of_nodes(),
                        1
                    )

                ]



                rows.append(row)







        arr = np.asarray(

            rows,

            dtype=np.float32

        )





        output = np.zeros(

            (

                self.k*self.m,

                BASE_FEATURES

            ),

            dtype=np.float32

        )





        if len(arr):


            output[

                :

                min(

                    len(arr),

                    len(output)

                )

            ] = arr[

                :

                len(output)

            ]





        return output.flatten()





    # --------------------------------------------------
    # Dimension
    # --------------------------------------------------


    @property

    def dimension(self):


        return len(

            self.vector()

        )







# ==================================================
# Helper Functions
# ==================================================



def nx_ego_nodes(

    G,

    center,

    radius

):


    return nx.single_source_shortest_path_length(

        G,

        center,

        cutoff=radius

    ).keys()





def nx_shortest_hops(

    G,

    source,

    target

):


    return nx.shortest_path_length(

        G,

        source,

        target

    )





def get_node_edges(

    G,

    node

):


    if isinstance(

        G,

        nx.MultiDiGraph

    ):


        return list(

            G.out_edges(

                node,

                keys=True,

                data=True

            )

        )



    else:


        return list(

            G.edges(

                node,

                keys=True,

                data=True

            )

        )






# ==================================================
# External Interface
# ==================================================


def neighborhood_state(
    G,
    source,
    destination,
    k=K,
    m=M,
    radius=2,
    transaction=None,
    candidate_paths=None,
    simulation_info=None,
    bucket_info=None
):
    """
    Public interface for generating RL observation.

    Used by:

    - Evaluations
    - PPO Agent
    - Routing modules


    Returns:
        numpy.ndarray
    """



    state = State(

        G,

        source,

        destination,

        transaction=transaction,

        candidate_paths=candidate_paths,

        simulation_info=simulation_info,

        bucket_info=bucket_info,

        k=k,

        m=m,

        radius=radius

    )



    return state.vector()