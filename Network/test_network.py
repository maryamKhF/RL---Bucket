from Network.graph_builder import LNGraphBuilder
from Network.environment import LightningEnvironment


# =====================================
# 1. Create synthetic Lightning Network
# =====================================

builder = LNGraphBuilder()

G = builder.synthetic(
    n=50,
    extra_edges=100,
    seed=42
)


print("===== NETWORK INFO =====")

print(
    "Nodes:",
    G.number_of_nodes()
)

print(
    "Channels:",
    G.number_of_edges()
)


# =====================================
# 2. Check Nodes
# =====================================

print("\n===== NODE SAMPLE =====")

for node_id, data in list(G.nodes(data=True))[:3]:

    print(
        node_id,
        data
    )



# =====================================
# 3. Check Channels
# =====================================

print("\n===== CHANNEL SAMPLE =====")


for u,v,k,data in list(
        G.edges(
            keys=True,
            data=True
        )
)[:3]:

    print(
        u,
        "->",
        v,
        "channel:",
        k
    )

    print(data)



# =====================================
# 4. Create Environment
# =====================================


env = LightningEnvironment(G)



# =====================================
# 5. Test neighbors
# =====================================

print("\n===== NEIGHBORS =====")


source = list(
    G.nodes()
)[0]


print(
    "Source:",
    source
)


print(
    env.get_neighbors(source)
)



# =====================================
# 6. Test Payment
# =====================================


nodes = list(
    G.nodes()
)


path = [

    nodes[0],
    nodes[1]

]


print("\n===== PAYMENT TEST =====")


result, info = env.execute_payment(
    path,
    amount=1000
)


print(
    "Result:",
    result
)


print(
    "Info:",
    info
)



# =====================================
# 7. Statistics
# =====================================

print("\n===== STATISTICS =====")

print(
    env.statistics()
)