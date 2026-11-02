                                                                                                                                                                                                    2025 7th International Conference on Blockchain Computing and Applications (BCCA)




                                                                                                                                                                              Bucket: Backtrack Routing for the Lightning
                                                                                                                                                                                               Network
2025 7th International Conference on Blockchain Computing and Applications (BCCA) | 979-8-3315-0296-6/25/$31.00 ©2025 IEEE | DOI: 10.1109/BCCA66705.2025.11229758




                                                                                                                                                                                                                      Amin Bashiri and Majid Khabbazian
                                                                                                                                                                                                               Department of Electrical and Computer Engineering
                                                                                                                                                                                                                        University of Alberta, Canada


                                                                                                                                                                       Abstract—The Lightning Network (LN) is a pioneering pay-                       of intermediary balances. Consequently, if a channel along
                                                                                                                                                                    ment channel network developed to address Bitcoin’s scalability                   the path lacks sufficient balance to forward the payment—a
                                                                                                                                                                    challenges. In LN, source nodes select transaction paths without                  situation more likely for larger payments and longer paths—
                                                                                                                                                                    knowing intermediary channel balances as channel balances are
                                                                                                                                                                    known only to their owners. Consequently, payments often fail                     the payment attempt fails. An error message is then propagated
                                                                                                                                                                    when channels lack sufficient funds, forcing the source to retry                  all the way back to the source, which must retry the payment
                                                                                                                                                                    the entire transaction.                                                           using a new path. This trial-and-error process increases latency
                                                                                                                                                                       This paper introduces Bucket, a novel atomic-payment solution                  and degrades the user experience.
                                                                                                                                                                    designed to reduce latency by minimizing these retries without
                                                                                                                                                                                                                                                         Currently, LN exhibits high payment latency: the 90th-
                                                                                                                                                                    requiring nodes to share their channel balances or the source
                                                                                                                                                                    node to lock additional funds. Bucket achieves this by routing                    percentile latency is around 30 seconds, and the 95th-percentile
                                                                                                                                                                    multiple independent payment alternatives simultaneously in a                     latency reaches 3 minutes [11], with 5% of payments taking
                                                                                                                                                                    “bucket,” each with its own route. Intermediate nodes decrypt all                 even longer. These delays highlight the need for more efficient
                                                                                                                                                                    alternatives, select a viable next hop based on their local channel               routing strategies.
                                                                                                                                                                    balance knowledge, group relevant alternatives, and forward them
                                                                                                                                                                                                                                                         Bucket is a method introduced in this work to significantly
                                                                                                                                                                    in a new bucket. If a chosen route becomes blocked, Bucket
                                                                                                                                                                    supports efficient backtracking by enabling immediate previous                    reduce payment latency while requiring minimal changes to the
                                                                                                                                                                    nodes (not necessarily the source) to quickly select alternative                  LN protocol and existing implementations. Bucket addresses
                                                                                                                                                                    paths. Simulation results using real LN data demonstrate that                     the root cause of high latency by avoiding full re-attempts
                                                                                                                                                                    Bucket significantly reduces latency and improves transaction                     when a path fails due to insufficient balance in an intermediary
                                                                                                                                                                    success rates, especially in challenging routing scenarios.
                                                                                                                                                                       Index Terms—Bitcoin, Lightning Network, Routing
                                                                                                                                                                                                                                                      channel. Instead, it enables early and efficient identification
                                                                                                                                                                                                                                                      of viable routes and rerouting of payments by intermediary
                                                                                                                                                                                             I. I NTRODUCTION                                         nodes. Simulations using real-world LN data show that Bucket
                                                                                                                                                                       A payment channel allows two parties to transact off-chain,                    achieves substantial improvements—reducing 95th-percentile
                                                                                                                                                                    with blockchain interaction required only at channel creation,                    latency by approximately 76% and 90th-percentile latency by
                                                                                                                                                                    closure, or in the event of a dispute [16]. To create a channel,                  around 69%.
                                                                                                                                                                    one or both parties publish an initial funding transaction on                        Bucket is fully compatible with source-based routing,
                                                                                                                                                                    the blockchain, locking funds into the channel [2]. The total                     the approach used by major LN implementations such as
                                                                                                                                                                    amount of locked funds is called the channel capacity, while                      c-lightning [7] and LND [20], in which the sender de-
                                                                                                                                                                    the current distribution of these funds between the parties is                    termines the complete payment path. It preserves key privacy
                                                                                                                                                                    referred to as the channel balance. This balance changes as the                   guarantees: the source does not reveal the destination’s identity
                                                                                                                                                                    parties pay each other. When the channel is closed, the most                      (which remains encrypted), and intermediary nodes are not
                                                                                                                                                                    up-to-date balance is published to the blockchain to settle the                   required to disclose their channel balances.
                                                                                                                                                                    final distribution of funds.                                                         Moreover, Bucket avoids the need for the sender to lock up
                                                                                                                                                                       Multiple payment channels can be linked together to form                       more funds than the actual payment amount or to consume more
                                                                                                                                                                    a Payment Channel Network (PCN), enabling multi-hop pay-                          Hash Time-Locked Contract (HTLC) slots—a scarce network
                                                                                                                                                                    ments that traverse one or more intermediaries to reach the final                 resource—than required by standard single-path routing. It also
                                                                                                                                                                    recipient [15], [24], [37], [5], [38]. The most prominent PCN                     preserves the atomicity of payments.
                                                                                                                                                                    today is the Lightning Network (LN) [29], deployed on Bitcoin.
                                                                                                                                                                                                                                                                                II. R ELATED W ORK
                                                                                                                                                                       LN employs onion routing to protect privacy: the source node
                                                                                                                                                                    selects all or most of the payment path [3], [4] and constructs                      The payment failure rate in LN, particularly for larger
                                                                                                                                                                    an onion-encrypted packet for transmission. This ensures that                     payments, has been a persistent concern. This section examines
                                                                                                                                                                    neither intermediaries nor observers can identify the source or                   past attempts at mitigating this issue and highlights Bucket’s
                                                                                                                                                                    destination of the payment.                                                       advantages and uniqueness.
                                                                                                                                                                       Another defining characteristic of LN is the privacy of chan-                     Improvements to Source Routing. Numerous studies have
                                                                                                                                                                    nel balances. Channel owners do not disclose their balances, so                   attempted to enhance the routing algorithm of source-routed
                                                                                                                                                                    when a source node selects a path, it does so without knowledge                   onion packets [30], [32], [23], [33], [19], [39], [26], [41],




                                                                                                                                                                           Authorized licensed use limited to: Vrije Universiteit Amsterdam. Downloaded on July 10,2026 at 10:24:23 UTC from IEEE Xplore. Restrictions apply.
                                                                                                                                                                       979-8-3315-0296-6/25/$31.00 ©2025 IEEE                                                                                                                   218
                                2025 7th International Conference on Blockchain Computing and Applications (BCCA)




[18], [21]. These works do not leverage the intermediary nodes                    Figure 1. Suppose that node A wishes to send a payment to
knowledge of channel balances and do not provide backtrack-                       node E. A can choose either A → B → C → E or A → B →
ing. Bucket does not overlap with these works as we propose                       D → E for this purpose. Sending the payment through the
a novel method to facilitate backtrack routing. Although the                      above paths requires A to create onion packets O1 : B(C(E))
payment is still source-routed, the intermediary nodes will have                  and O2 : B(D(E)) respectively.
influence in changing the path. All the methods mentioned
above can perfectly complement Bucket.
   Splitting Payments. A considerable amount of research has
focused on proposing methods to split payments, particularly
large payments, into smaller ones to utilize multiple paths
with low balances for routing [1], [28], [36], [12], [25], [17].
However, unlike Bucket, these methods are inefficient in their
consumption of HTLC slots. Moreover, due to atomicity, the
failure of any single part results in the failure of the entire                                 Fig. 1. The channels graph of a sample network.
payment. Consequently, as demonstrated in [31], these methods
may not effectively reduce latency. While relaxing the atomicity
constraint [13] can mitigate this issue, most real-world use cases
require atomicity to ensure reliable transactions.
   Adding Redundancy. Another approach involves introduc-
ing redundancy to payments [6], [31], [35] and splitting them
across multiple paths. While effective in reducing payment                               Fig. 2. Source node A sends a payment to the destination node
                                                                                         E through the intermediary node C.
failure rates and latencies, these methods require the payer
to lock up funds that are several times the original payment
amount and are inefficient in utilizing the limited HTLC slots.
Similar to the previously mentioned methods, if needed, these
methods can complement Bucket by adding redundancy to
payments, with each payment routed using backtrack routing
to achieve a higher success rate.
   Local Routing. Local routing approaches [14], [33] allow
intermediary nodes to actively participate in routing decisions                          Fig. 3. Source node A sends a payment to the destination node
and can support mechanisms such as backtracking, as used in                              E through the intermediary node D.
Bucket. While both approaches improve payment success rates
and preserve privacy, they introduce notable drawbacks and
trade-offs.
   First, both rely heavily on multi-path routing, which in-
creases the number of required HTLCs in the network. Given
the strict per-channel HTLC limits in LN, this can lead to
HTLC exhaustion and make the network more vulnerable to                                  Fig. 4. Source node A sending a bucket containing multiple
congestion attacks [27], [22].                                                           onion packets through the intermediary node B.
   Second, these protocols require the creation and maintenance
of spanning trees to support coordinate-based or structured                          Since A does not have information about the balances of
routing. This adds synchronization and messaging overhead,                        channels B → C and B → D, irrespective of which of the
particularly in dynamic environments where channel balances                       above two paths node A selects, A’s initial attempt may fail,
change frequently. Although SpeedyMurmurs reduces this over-                      in which case A has to try the alternative path: If A chooses to
head through on-demand stabilization, the maintenance burden                      proceed through node C, it will forward the onion packet O1 :
remains a scalability concern.                                                    B(C(E)) as illustrated in Figure 2. If the channel B → C does
                                                                                  not have enough balance to forward the payment, it will fail,
                     III. S OLUTION OVERVIEW                                      and an error will be propagated back to the source. The source
   At the high level, the proposed method involves sending a                      node can then re-attempt the payment with the alternative path
group of onion packets for a single payment, each created to                      through node D and send the O2 onion packet as shown in
travel on a different path. This allows the intermediary nodes                    Figure 3. While functional, this sequential retry process can
to choose the best one to forward and then try the other ones                     be time-consuming and inefficient, resulting in a degraded user
if the first try fails instead of sending an error all the way back               experience.
to the source to repeat the whole process.                                           Bucket seeks to address these inefficiencies without requiring
   Example. Let us explain the functionality of Bucket in a                       nodes to reveal their balance information. This is done by
simple example. For this, consider the network illustrated in                     allowing multiple onion packets to be forwarded simultaneously




       Authorized licensed use limited to: Vrije Universiteit Amsterdam. Downloaded on July 10,2026 at 10:24:23 UTC from IEEE Xplore. Restrictions apply.
                                                                                                                                                            219
                                2025 7th International Conference on Blockchain Computing and Applications (BCCA)




for a single payment without initially requiring the source node                  payment. As each HTLC locks up at most amt, the total amount
to use more funds than the original payment amount or consume                     of funds locked across the network is bounded by L · amt.
more HTLCs than needed for a regular payment at any given
                                                                                     We note that Bucket generalizes the standard single-path
time.
                                                                                  payment model in LN. A regular payment without rerouting
   Sending the payment using Bucket is illustrated in Figure 4.
                                                                                  or backtracking is simply a special case of Bucket, where
As shown in the figure, Node A places the two onion packets
                                                                                  only a single path is used and no additional behavior is
O1 : B(C(E)) and O2 : B(D(E)) in a bucket and forwards
                                                                                  triggered. In such cases, Bucket behaves identically to the
the bucket to node B. The intermediary node B, knowing the
                                                                                  existing implementation—using a single onion packet and a
balances of its channels with nodes C and D, can select one of
                                                                                  single HTLC per hop.
the two onion packets and forward it to the next node. There’s
                                                                                     Thus, using Bucket is entirely optional for each payment.
still a risk of failure since node B is not aware of the balances
                                                                                  A source node may choose to send a standard single-path
of channels C → E and D → E. However, the source node
                                                                                  payment, incurring no additional overhead. However, from the
A can be sure that between B → D or B → C, the capable
                                                                                  sender’s perspective, Bucket offers a strong incentive to adopt
channel will be chosen by node B. This is because node B
                                                                                  the method due to its lower latency. Similarly, intermediary
now has the flexibility to make the best decision based on its
                                                                                  nodes are motivated to participate, as it increases their like-
knowledge of local channel balances.
                                                                                  lihood of successfully forwarding the payment and collecting
   Furthermore, Bucket allows backtracking: Suppose there is                      fees.
enough balances on both channels B → D or B → C, and
that B forwards O1 to C. If C → E channel happens to lack                                                  IV. I MPLEMENTATION
adequate balance, node C will return an error onion packet                           In LN, a node A forwards a payment to the next node
to node B. Recognizing that the payment via node C failed,                        B by sending a so called update_add_htlc message.
node B can then attempt to send O2 instead of returning the                       Table I shows the format of an update_add_htlc message.
error back to node A and requiring the re-attempt of the whole                    The message includes all necessary details for the forwarding
process.                                                                          process, such as the channel to use, the payment amount, the
   We note that intermediary nodes choose only one of the                         timeout for settlement, and an onion packet containing routing
neighbors to forward the payment to at any given time, and if                     instructions.
the payment through that neighbor fails, they will try another
                                                                                    Field                      Size             Description
neighbor. In this case, the HTLC in the B → C channel is freed
                                                                                    type                       2 bytes          Message type (128 for
before creating an HTLC in the B → D channel. Therefore,                                                                        update_add_htlc).
Bucket does not consume more HTLC slots than a regular                              channel_id                 32 bytes         Unique identifier for the chan-
payment at any given time.                                                                                                      nel.
                                                                                    id                         8 bytes          Unique identifier for this
   To maintain privacy in Bucket, all onion packets transferred                                                                 HTLC, assigned by the
through the network will have the same size and appear distinct                                                                 sender.
at any stage, similar to the current implementation.                                amount_msat                8 bytes          Amount to forward in mil-
                                                                                                                                lisatoshis (1 sat = 1000 msat).
   Unlike methods that introduce redundancy or multi-path                           payment_hash               32 bytes         SHA256 hash of the payment
splitting, Bucket does not require the source node to lock                                                                      preimage for HTLC resolution.
up more funds and consume more HTLC slots than standard                             cltv_expiry                4 bytes          The     CLTV       (CheckLock-
                                                                                                                                TimeVerify) value for this
single-path payments. These properties are formally stated in                                                                   HTLC.
the following proposition.                                                          onion_packet               1366 bytes       Full onion packet with routing
                                                                                                                                information for the payment.
Proposition 1. Let L denote the length of the longest single                                                         TABLE I
payment path used by Bucket, and let amt denote the payment                                 T HE FORMAT OF U P D A T E _ A D D _ H T L C M ESSAGES IN LN.
amount. Then, at any point in time, the number of outstanding
HTLCs used by Bucket is at most L, and the total amount of
funds locked in the network is bounded by L · amt.                                   Upon receiving the message, the node first ensures that the
                                                                                  request is valid and compatible with the current state of its
Proof. At any point in time, each intermediary node in Bucket                     channel with the sender. This validation involves checking
forwards the payment to at most one of its neighbors. For-                        whether the channel has sufficient capacity to accommodate
warding to multiple neighbors simultaneously would expose                         the payment and whether the provided timeout is adequate for
the node to financial risk, as it may be unable to claim the                      successful processing. If the request fails these checks, the node
corresponding incoming HTLCs if only one of the outgoing                          rejects the HTLC and promptly notifies the sender of the failure.
ones is fulfilled.                                                                   If the HTLC passes validation, the node processes the in-
   Therefore, the set of outstanding HTLCs at any moment                          cluded onion packet to determine the next step in the route. The
forms a prefix of a single payment path, rather than a branching                  routing instructions specify the payment amount to forward, the
structure. Since the maximum path length in Bucket is L, there                    timeout for the next hop, and the channel to use. Using this
can be at most L outstanding HTLCs associated with a given                        information, the node prepares a new update_add_htlc




       Authorized licensed use limited to: Vrije Universiteit Amsterdam. Downloaded on July 10,2026 at 10:24:23 UTC from IEEE Xplore. Restrictions apply.
                                                                                                                                                            220
                                2025 7th International Conference on Blockchain Computing and Applications (BCCA)




message to send to the next node. Simultaneously, the node                        ment alternatives within the same Bucket, it evaluates their
updates its internal channel state to include the new HTLC,                       next hops. If an alternative targets a next node X with which
ensuring that the payment is securely tracked until it is either                  node B already has an outstanding HTLC for another payment
completed or fails. The node then proceeds with the forward-                      alternative in the same Bucket, node B immediately forwards
ing process by transmitting the updated update_add_htlc                           the payment to node X. If no outstanding HTLC exists with
message to the next hop, progressing the payment along the                        node X, node B stores the alternative locally. This allows for
route.                                                                            quick re-routing decisions if earlier attempts fail.
   How Bucket Works. Bucket was designed with simplicity                             Case study. To illustrate the effectiveness of Bucket in
of implementation in mind. In Bucket, node A forwards each                        reducing payment latency, consider the network shown in
payment alternative using a standard update_add_htlc                              Figure 5, where node A intends to send a payment to node
message, with a minor adjustment to indicate that the message                     E. There are six possible paths available, shown explicitly in
is part of a Bucket. Initially, one might consider using the same                 the figure. Suppose that channels B → G, C → E, B → D,
id for multiple payments within a Bucket to logically group                       C → G, and C → F do not have enough balance to forward
them. However, according to the LN specification (BOLT #2),                       the payment. Without knowledge of these channel balances, the
each HTLC in a channel must have a unique id. Thus, using                         conventional sequential routing approach used by LN requires
identical IDs for multiple HTLCs within the same channel is                       node A to try these paths one by one. This means node A must
prohibited and would result in rejection of these payments by                     attempt each path until discovering that the first five paths fail,
compliant nodes.                                                                  only succeeding on the sixth path.
   To maintain full compatibility with the LN protocol, Bucket                       With Bucket, instead of sequential attempts, node A si-
instead assigns a unique id to each HTLC, even when forward-                      multaneously constructs and forwards six onion packets cor-
ing multiple alternatives in the same Bucket. Logical grouping                    responding to these paths within a single Bucket to node
of these alternatives can still be accomplished by encoding                       B. Node B processes each packet as it arrives, immediately
additional metadata within the existing onion_packet field.                       dropping those that cannot be forwarded due to insufficient
Specifically, Bucket leverages the extensible Type-Length-                        channel balance. Specifically, upon receiving the first packet
Value (TLV) records supported by the Lightning protocol                           destined for node G, node B drops it due to insufficient funds
(BOLT #4) to embed a custom-defined bucket_id. Inter-                             on the B → G channel. The second packet, targeting node
mediary nodes that support Bucket recognize this TLV field                        C, is successfully forwarded by node B. The third packet is
and use the bucket_id to group related HTLCs, enabling                            similarly dropped because channel B → D lacks sufficient
efficient forwarding decisions without compromising protocol                      funds. However, packets four, five, and six, all intended for
compliance.                                                                       node C, are forwarded immediately since an outstanding HTLC
   Additionally, nodes can signal support for Bucket function-                    with node C already exists.
ality using feature bits during channel negotiation as specified
                                                                                     Out of the four packets node C receives (packets 2, 4, 5, and
in BOLT #9. By introducing a custom optional feature bit,
                                                                                  6), it forwards only packet 6 successfully, as channels C → E,
nodes explicitly advertise their ability to handle Bucket-marked
                                                                                  C → G, and C → F lack adequate balance. Thus, node D
HTLCs. These signaling approaches ensure seamless integra-
                                                                                  receives a single packet from node C, which it forwards through
tion with existing LN implementations, promoting adoption
                                                                                  node F to the final destination, node E.
without necessitating significant protocol modifications.
   When node B receives the first payment from a Bucket,                             We remark that Bucket does not require the sender to include
it processes and forwards the payment to the next node, say                       an onion packet for every possible route. Instead, the sender
node C, as usual. Each payment alternative in a Bucket has a                      selects a small subset of promising paths—typically those
unique id, but node B recognizes related payments through                         identified by its existing routing heuristic—and places only
the embedded bucket_id encoded in the TLV record within                           these packets into a Bucket.
the onion_packet. When B subsequently receives another                               Latency. For latency analysis, assume each message trans-
payment alternative belonging to the same Bucket, it examines                     mission or HTLC update between nodes takes α seconds. Under
the next node indicated by the routing information.                               the sequential routing method, node A must attempt each path
   If the subsequent payment is also intended for node C, node                    in order, accumulating latencies of 2α for packet 1, 4α for
B immediately processes and forwards it accordingly. However,                     packet 2, 2α for packet 3, 4α for packet 4, 4α for packet 5,
if the alternative specifies a different next node, say node D,                   and 5α for packet 6. This results in a total cumulative latency
node B temporarily holds this alternative for potential future                    of 21α.
use. Should node C later notify node B of its inability to                           In contrast, Bucket’s simultaneous forwarding reduces la-
complete the payment (due to insufficient channel balance or                      tency substantially. As node B and subsequent nodes quickly
other issues), node B can then forward the previously held                        discard unviable packets without needing sequential retries, the
alternative to node D without involving the source node, thus                     latency experienced corresponds only to the successful packet,
avoiding unnecessary payment retries.                                             resulting in a total latency of just 5α. Thus, Bucket significantly
   More generally, whenever node B receives additional pay-                       enhances efficiency by reducing the overall payment latency.




       Authorized licensed use limited to: Vrije Universiteit Amsterdam. Downloaded on July 10,2026 at 10:24:23 UTC from IEEE Xplore. Restrictions apply.
                                                                                                                                                            221
                                2025 7th International Conference on Blockchain Computing and Applications (BCCA)




                                         Fig. 5. Six examples of possible paths node A can take to reach node E.



                    V. S IMULATION R ESULTS                                       six years1 . Consequently, in our simulation, we randomly
                                                                                  saturate each channel to one side based on this information.
   We conduct simulations to demonstrate Bucket’s effective-                         In Section IV, we introduced the variable α to represent the
ness in reducing payment latency. Previous studies have used                      latency of forwarding an onion packet over one hop, including
different types of network graphs for simulations. Some [30],                     the time required to update the commitment transactions or
[39] have employed synthetic graphs like the Watts-Strogatz                       propagate an error message to the previous hop followed by a
model [40], while others have used real-world graph ex-                           commitment update. While previous works have assumed this
ports [14]. We adopt the latter approach to get more accurate                     value to be 100ms [14], we find this assumption unrealistic, as
results and run our simulations on a snapshot of the real-world                   updating commitment transactions typically requires multiple
LN graph, exported on May 20, 2024. At that time, the network                     round trips. Since our simulation compares the current imple-
contained 15,120 nodes and 51,579 channels.                                       mentation with Bucket, we continue to employ multiples of α
   One of the primary challenges in simulating LN is the lack                     to represent the time required for processing each transaction.
of publicly available data for many of its critical attributes.                      Bucket is recommended for transactions that are more diffi-
Key examples include channel balances, the proportion of                          cult to route through the network because of the harsh environ-
offline or unavailable nodes, and the amount distribution and                     ment, a high payment amount, or a long path. Bucket’s benefits
frequency of payments within the network. Where possible, we                      become increasingly apparent as payment amounts increase or
base our simulation parameters on established studies in the                      when longer routes are needed. Smaller payment amounts are
field. In cases where such data is unavailable, we adopt well-                    generally more likely to succeed, so they are not expected to use
founded assumptions to enhance the realism of the simulation                      Bucket. However, as payment amounts grow, Bucket offers a
environment.                                                                      substantial improvement in reducing latency. In this simulation,
   Regarding channel capacities, while some prior studies have                    all transactions will transfer 105 satoshis (0.001 Bitcoin), which
employed synthetic data [14], we use real-world capacities,                       we consider a reasonable threshold for utilizing Bucket.
as this information is publicly available. However, public                           The maximum number of hops is capped at 20. This limit
data is lacking for channel balances, the capacity distribution                   is determined by the fixed size of the routing_info field,
between channel participants, requiring us to make informed                       which is 1300 bytes in the onion routing packet, and the fact
assumptions. Prior research suggests that when the probability                    that each hop payload requires 65 bytes. Based on this, the
of payments flowing in each direction across a channel is not                     maximum number of hops is calculated as:
exactly 0.5, channels tend to saturate quickly [9], [34], [8], with
most or all capacity accumulating on one side. Furthermore,                         1 Channel age data was sourced from 1ml.com. While this data may not be
publicly available data indicates that the median channel age in                  entirely accurate, it provides a reasonable basis for our assumptions, even if
the real-world LN is one year, with the 95th percentile reaching                  only approximately reflective of real-world conditions.




       Authorized licensed use limited to: Vrije Universiteit Amsterdam. Downloaded on July 10,2026 at 10:24:23 UTC from IEEE Xplore. Restrictions apply.
                                                                                                                                                            222
                                2025 7th International Conference on Blockchain Computing and Applications (BCCA)




                  Routing information size (1300 bytes)
  Max hops =                                            = 20
                    Payload size per hop (65 bytes)
   For each routing method, we limit the maximum number
of attempts or paths to 255 per payment. We assume that
paths will include a minimum of four intermediary hops to
ensure adequate privacy. While this assumption can vary, as
different source nodes may require distinct trade-offs between
privacy and latency, we consider four intermediary hops to be
a reasonable baseline.
   To make the simulation environment more realistic, we
randomly mark 20 percent of the nodes as offline, as this has
been observed for LN before [10]. If a payment attempts to
pass through these offline nodes, it will fail and send an error                  Fig. 6. Simulation results showing transaction latency improvements across
onion packet back to the source. However, if these offline nodes                  different percentiles.
are selected as either the source or destination of a payment,
they will function normally, as we assume that the source and
destination of legitimate payments will not be offline.                           to the average, making the network more reliable and suitable
                                                                                  for widespread use.
   We start the simulation by randomly selecting nodes from
the entire pool to generate a list of source and destination                        Latency Percentile      Source Routing (α)       Bucket (α)      Improvement
pairs. After each selection, we return the nodes to the pool                              25th                      5                    4              20%
before choosing another pair. Transactions between these pairs                            50th                      9                    5             44.44%
                                                                                          75th                      22                   9             59.09%
are executed in two separate simulations using different routing                          90th                      59                  18             69.49%
methods. Since the source and destination pairs and the initial                           95th                     112                  27             75.89%
graph are the same for both, and each tries up to 255 different                                              TABLE II
paths per payment, the success rate will be identical but with                       L ATENCY COMPARISON ACROSS PERCENTILES BETWEEN S OURCE
different latencies.                                                               ROUTING AND B UCKET, ALONG WITH THE IMPROVEMENT PERCENTAGE .
   For each routing method, we iterate through the source and
destination pairs. A transaction is attempted only if the source                                         VI. C ONCLUSION
and destination have sufficient balances in at least one of their                    We introduced Bucket, a novel routing approach that enables
channels. This condition is essential because the nodes know                      intermediary nodes to re-attempt payments through alternative
their channel balances and will only initiate a transaction if they               paths while allowing the source node to retain overall control
satisfy the payment amount. If the conditions are unmet, we                       of the process. With Bucket, the source node sends multiple
skip the source-destination pair without counting it as neither                   onion packets simultaneously within a virtual bucket. This
a success nor a failure and proceed to the next pair without                      allows intermediary nodes to decide which node to forward
recording any latency. This process continues until we attempt                    the payment to based on their knowledge of channel balances.
ten thousand transactions on each simulation.                                     Additionally, it allows nodes to backtrack and explore alterna-
   Figure 6 illustrates the simulation results using box plots.                   tive paths without sending error messages back to the source,
To enhance clarity, the data in this figure excludes outliers. For                eliminating the need to restart the payment process.
completeness, we included all the simulation results in Table II.                    What sets Bucket apart from existing methods is that it
This table provides a detailed comparison of transaction latency                  does not lock up additional funds or consume more HTLC
at various percentiles.                                                           slots than a regular single payment, does not require nodes to
   The data presented in Table II demonstrates the substantial                    disclose the balance information of their channels, and ensures
impact of Bucket, particularly for a global payment system like                   that the destination node’s identity remains encrypted. Bucket
LN, where average latency alone does not sufficiently capture                     requires minimal modifications to the existing protocol, yet our
the performance improvements. In a payment network striving                       results demonstrate its ability to significantly reduce latency,
for broad adoption in everyday transactions, the true measure                     particularly in challenging routing scenarios, such as those
of a routing system’s effectiveness lies in its performance under                 involving large payment values. Simulation results show that
worst-case conditions. As the saying goes, a system is only as                    Bucket decreases the likelihood of users experiencing extreme
strong as its weakest link, or in this case, its highest latency                  delays. This enhances the network’s reliability and making it
percentiles. The improvements observed in the 90th and 95th                       more suitable for widespread adoption. Finally, Bucket comple-
percentiles are especially noteworthy, highlighting how Bucket                    ments existing payment methods, such as multi-part payments,
significantly enhances performance in those critical cases. With                  and can be combined with them to further improve success
Bucket, fewer users will experience extreme delays compared                       rates and reduce latency.




       Authorized licensed use limited to: Vrije Universiteit Amsterdam. Downloaded on July 10,2026 at 10:24:23 UTC from IEEE Xplore. Restrictions apply.
                                                                                                                                                            223
                                                                     2025 7th International Conference on Blockchain Computing and Applications (BCCA)




                                                                  R EFERENCES                                              Symposium on Modeling and Optimization in Mobile, Ad hoc, and
                                                                                                                           Wireless Networks (WiOpt). pp. 193–200 (2022)
                                    [1] AMP: Atomic multi-path payments over lightning.                               [22] Lu, Z., Han, R., Yu, J.: General congestion attack on htlc-based pay-
                                        https://lists.linuxfoundation.org/pipermail/                                       ment channel networks. In: Proceedings of the International Confer-
                                        lightning-dev/2018-February/000993.html,                                           ence on Blockchain Economics, Security and Protocols (Tokenomics)
                                        Accessed: Dec. 23rd, 2024                                                          (2021), https://arxiv.org/abs/2103.06689, also available as
                                    [2] [lightning-dev] dual funding proposal.                                             arXiv:2103.06689
                                        https://lists.linuxfoundation.org/pipermail/                                  [23] Malavolta, G., Moreno-Sanchez, P., Kate, A., Maffei, M.: Silentwhispers:
                                        lightning-dev/2018-November/001682.html,                                           Enforcing security and privacy in decentralized credit networks. In:
                                        Accessed: Dec. 23rd, 2024                                                          Annual Network and Distributed System Security Symposium (2017)
                                    [3] Rendez vous mechanism on top of sphinx.                                       [24] Malavolta, G., Moreno-Sanchez, P., Schneidewind, C., Kate, A., Maffei,
                                        https://github.com/lightning/bolts/wiki/                                           M.: Anonymous multi-hop locks for blockchain scalability and interoper-
                                        Rendez-vous-mechanism-on-top-of-Sphinx,                                            ability. In: Annual Network and Distributed System Security Symposium
                                        Accessed: Dec. 23rd, 2024                                                          (2019)
                                                                                                                      [25] Mazumdar, S., Ruj, S.: Cryptomaze: Privacy-preserving splitting of off-
                                    [4] Route blinding.
                                                                                                                           chain payments. IEEE Transactions on Dependable and Secure Comput-
                                        https://github.com/lightning/bolts/
                                                                                                                           ing 20(2), 1060–1073 (2022)
                                        blob/master/proposals/route-blinding.md,
                                                                                                                      [26] Mazumdar, S., Ruj, S., Singh, R.G., Pal, A.: HushRelay: A privacy-
                                        Accessed: Apr. 3, 2025
                                                                                                                           preserving, efficient, and scalable routing algorithm for off-chain pay-
                                    [5] Aumayr, L., Moreno-Sanchez, P., Kate, A., Maffei, M.: Blitz: Secure                ments. In: IEEE International Conference on Blockchain and Cryptocur-
                                        Multi-Hop payments without Two-Phase commits. In: USENIX Security                  rency (ICBC). pp. 1–5 (2020)
                                        Symposium. pp. 4043–4060 (2021)                                               [27] Mizrahi, A., Zohar, A.: Congestion attacks in payment channel networks.
                                    [6] Bagaria, V.K., Neu, J., Tse, D.: Boomerang: Redundancy improves                    In: International conference on financial cryptography and data security.
                                        latency and throughput in payment-channel networks. In: Financial Cryp-            pp. 170–188. Springer (2021)
                                        tography and Data Security. Lecture Notes in Computer Science, vol.           [28] Piatkivskyi, D., Nowostawski, M.: Split payments in payment networks.
                                        12059, pp. 304–324 (2020)                                                          In: Springer International Workshop on Data Privacy Management, Cryp-
                                    [7] Blockstream: c-lightning (2018),                                                   tocurrencies and Blockchain Technology. pp. 67–75 (2018)
                                        https://github.com/ElementsProject/lightning,                                 [29] Poon, J., Dryja, T.: The bitcoin lightning network: Scalable off-chain
                                        accessed: 2025-04-03                                                               instant payments (2016)
                                    [8] Brânzei, S., Segal-Halevi, E., Zohar, A.: How to charge lightning: The       [30] Prihodko, P., Zhigulin, S., Sahno, M., Ostrovskiy, A., Osuntokun, O.:
                                        economics of bitcoin transaction channels. In: Allerton Conference on              Flare: An approach to routing in lightning network (2016)
                                        Communication, Control, and Computing (Allerton). pp. 1–8 (2022)              [31] Rahimpour, S., Khabbazian, M.: Spear: fast multi-path payment with
                                    [9] Dandekar, P., Goel, A., Govindan, R., Post, I.: Liquidity in credit net-           redundancy. In: ACM Conference on Advances in Financial Technologies
                                        works: A little trust goes a long way. In: ACM conference on Electronic            (AFT). pp. 183–191 (2021)
                                        commerce. pp. 147–156 (2011)                                                  [32] Roos, S., Beck, M., Strufe, T.: Anonymous addresses for efficient and
                                   [10] Decker, C.: C-lightning plugins 05: Probe plugin results.                          resilient routing in f2f overlays. In: IEEE International Conference on
                                        https://blog.blockstream.com                                                       Computer Communications (INFOCOM). pp. 1–9 (2016)
                                        /c-lightning-plugins-05-probe-plugin-results/,                                [33] Roos, S., Moreno-Sanchez, P., Kate, A., Goldberg, I.: Settling payments
                                        Accessed: Dec. 23rd, 2024                                                          fast and private: Efficient decentralized routing for path-based transac-
                                   [11] Decker, C.: Channels, flows and icebergs.                                          tions. arXiv preprint arXiv:1709.05748 (2017)
                                        youtu.be/HtU7ZlxvLL4?t=5810,                                                  [34] Shabgahi, S.Z., Hosseini, S.M., Shariatpanahi, S.P., Bahrak, B.: Modeling
                                        Accessed: Dec. 23rd, 2024                                                          effective lifespan of payment channels. arXiv preprint arXiv:2301.01240
                                   [12] Di Stasi, G., Avallone, S., Canonico, R., Ventre, G.: Routing payments             (2022)
                                        on the lightning network. In: IEEE international conference on internet       [35] Shen, Y., Ersoy, O., Roos, S.: Extras and premiums: Local pcn routing
                                        of things (IThings) and IEEE green computing and communications                    with redundancy and fees. In: International Conference on Financial
                                        (GreenCom) and IEEE cyber, physical and social computing (CPSCom)                  Cryptography and Data Security. pp. 110–127 (2023)
                                        and IEEE smart data (SmartData). pp. 1161–1170 (2018)                         [36] Sivaraman, V., Venkatakrishnan, S.B., Alizadeh, M., Fanti, G., Viswanath,
                                   [13] Dziembowski, S., Kedzior, P.: Non-atomic payment splitting in channel              P.: Routing cryptocurrency with the spider network. In: ACM Workshop
                                        networks. In: ACM Conference on Advances in Financial Technologies                 on Hot Topics in Networks. pp. 29–35 (2018)
                                        (AFT). vol. 282, pp. 17:1–17:23 (2023)                                        [37] Tripathy, S., Mohanty, S.K.: MAPPCN: Multi-hop anonymous and
                                   [14] Eckey, L., Faust, S., Hostáková, K., Roos, S.: Splitting payments locally        privacy-preserving payment channel network. In: International conference
                                        while routing interdimensionally. Cryptology ePrint Archive (2020)                 on financial cryptography and data security (FC). pp. 481–495 (2020)
                                   [15] Green, M., Miers, I.: Bolt: Anonymous payment channels for decen-             [38] Tsabary, I., Yechieli, M., Manuskin, A., Eyal, I.: MAD-HTLC: because
                                        tralized currencies. In: ACM SIGSAC conference on computer and                     HTLC is crazy-cheap to attack. In: IEEE Symposium on Security and
                                        communications security (CCS). pp. 473–489 (2017)                                  Privacy (SP). pp. 1230–1248 (2021)
                                   [16] Gudgeon, L., Moreno-Sanchez, P., Roos, S., McCorry, P., Gervais, A.:          [39] Wang, P., Xu, H., Jin, X., Wang, T.: Flash: efficient dynamic routing for
                                        SoK: Layer-two blockchain protocols. In: Financial Cryptography and                offchain networks. In: International Conference on Emerging Networking
                                        Data Security (FC). pp. 201–226 (2020)                                             Experiments And Technologies (CoNEXT). pp. 370–381 (2019)
                                   [17] Guo, J., Shang, L., Wang, Y., Liang, T., Wang, Z., An, H.: PAMP: A            [40] Watts, D.J., Strogatz, S.H.: Collective dynamics of ‘small-
                                        new atomic multi-path payments method with higher routing efficiency.              world’networks. nature 393(6684), 440–442 (1998)
                                        In: International Conference on Machine Learning for Cyber Security.          [41] Xue, H., Huang, Q., Bao, Y.: EPA-Route: Routing payment channel net-
                                        pp. 575–583 (2022)                                                                 work with high success rate and low payment fees. In: IEEE International
                                   [18] Hong, H.J., Chang, S.Y., Zhou, X.: Auto-Tune: Efficient autonomous                 Conference on Distributed Computing Systems (ICDCS). pp. 227–237
                                        routing for payment channel networks. In: IEEE Conference on Local                 (2021)
                                        Computer Networks (LCN). pp. 347–350 (2022)
                                   [19] Khalil, R., Gervais, A.: Revive: Rebalancing off-blockchain payment net-
                                        works. In: ACM SIGSAC conference on computer and communications
                                        security (CCS). pp. 439–453 (2017)
                                   [20] Lightning Labs: LND - Lightning Network Daemon (2017),
                                        https://github.com/lightningnetwork/lnd,                         accessed:
                                        2025-04-03
                                   [21] Liu, J., Chen, C., Zhou, L., Fang, Z.: Real-time recursive routing in
                                        payment channel network: A bidding-based design. In: IEEE International




                                           Authorized licensed use limited to: Vrije Universiteit Amsterdam. Downloaded on July 10,2026 at 10:24:23 UTC from IEEE Xplore. Restrictions apply.
                                                                                                                                                                                                224
Powered by TCPDF (www.tcpdf.org)

                                                    Engineering Applications of Arti cial Intelligence 146 (2025) 110225


                                                                  Contents lists available at ScienceDirect


                                        Engineering Applications of Artificial Intelligence
                                                       journal homepage: www.elsevier.com/locate/engappai


Research paper

Hybrid pathfinding optimization for the Lightning Network with
Reinforcement Learning
Danila Valko a,b              ,∗, Daniel Kudenko b
a
    OFFIS — Institute for Information Technology, Escherweg 2, 26121, Oldenburg, Lower Saxony, Germany
b
    L3S Research Center, Appelstr, 9a, 30167, Hannover, Lower Saxony, Germany



ARTICLE                 INFO                              ABSTRACT

Keywords:                                                 Payment channel networks, such as Bitcoin’s Lightning Network, have emerged to address blockchain
Payment channel networks                                  scalability issues, enabling rapid transactions. Despite their potential, these networks often experience payment
Payment success rate in Lightning Network                 failures due to delays in pathfinding, unreliable routes and infrastructure issues, resulting in excessive carbon
Pathfinding algorithms
                                                          emissions. Current reinforcement learning solutions for payment channel networks mainly address issues like
Dynamic optimization with Reinforcement
                                                          payment channel balance and routing fees but often overlook the infrastructure-related causes of payment
Learning
                                                          failure. This paper introduces a novel reinforcement learning-based architecture that combines reinforcement
                                                          learning agent with native deterministic pathfinding algorithms. This hybrid approach leverages the fast,
                                                          complete solutions of deterministic algorithms while adapting to the network’s dynamic and probabilistic
                                                          nature of payments to significantly enhance payment success rates. Experiments on real network snapshots
                                                          show that this approach outperforms native pathfinding algorithms and state-of-the-art static optimization
                                                          methods, providing improved reliability and efficiency in dynamic network conditions. In scenarios with
                                                          payment failure rates greater than 5%, the proposed approach achieves a 10% higher payment success
                                                          rate than existing methods, while maintaining balanced performance on economic key metrics such as the
                                                          payment fee and throughput, and on sustainability key metrics such as payment path length, number of
                                                          inter-country/continental hops, and average carbon intensity.



1. Introduction                                                                              SpeedyMurmurs, ProfitPilot, and Fence (Camilo et al., 2024; Wang
                                                                                             et al., 2024) as well as some machine learning architectures (Asgari
    Payment channel networks (PCNs), most notably the Lightning Net-                         et al., 2022; Papadis and Tassiulas, 2023; Song et al., 2024). Neverthe-
work (LN), have been widely adopted with the objective of facilitating                       less, none have been developed into comprehensive software solutions,
faster, lower-cost transactions off-chain. Nevertheless, they continue                       nor have they been implemented in LN software clients, due to practical
to encounter significant obstacles in attaining consistent success in                        limitations and the necessity for extensive online testing. Furthermore,
payment routing. A common criticism of PCNs, and LN in partic-                               the efficacy of these solutions has been assessed on relatively modest
ular, is the inconsistency in payment success rates due to network                           network samples, which presents a challenge in demonstrating their
architecture limitations and a lack of insight into the causes of pay-                       superiority over native algorithms in real-world scenarios (Chen et al.,
ment failures (Dasaklis and Malamas, 2023). For example, research                            2022b). It is notable that the current developments of determinis-
conducted by River revealed that payment success rates in their LN
                                                                                             tic algorithms (e.g. Flash) demonstrate inferior performance within a
infrastructure ranged from 96.5% to 99.6% (River, 2023). However,
                                                                                             realistic setup in terms of payment success compared to the actual
other reports have indicated much lower success rates, with instances
                                                                                             statistics of the LN with its native algorithms (see River (2023)). Con-
as low as 50% (Stadelmann, 2023). These failures frequently originate
                                                                                             sequently, the identification of a more robust and scalable pathfinding
from complications associated with pathfinding delays, the absence
                                                                                             solution represents a significant challenge in the pursuit of enhancing
of optimal routes, node unreliability, and payment channel balancing
                                                                                             the payment success rates in LN.
issues.
    A plethora of algorithms have been put forth with the aim of                                 This paper addresses the aforementioned issues of pathfinding and
enhancing payment routing and payment path finding in LN. These                              payment success by employing a novel approach based on reinforce-
include deterministic methods such as Flash, Flare, SilentWhispers,                          ment learning (RL). In contrast to existing RL-based solutions, which


     ∗ Corresponding author at: OFFIS — Institute for Information Technology, Escherweg 2, 26121, Oldenburg, Lower Saxony, Germany.
       E-mail addresses: danila.valko@offis.de (D. Valko), kudenko@l3s.de (D. Kudenko).

https://doi.org/10.1016/j.engappai.2025.110225
Received 23 June 2024; Received in revised form 17 January 2025; Accepted 3 February 2025
Available online 13 February 2025
0952-1976/© 2025 Elsevier Ltd. All rights are reserved, including those for text and data mining, AI training, and similar technologies.
D. Valko and D. Kudenko                                                                              Engineering Applications of Arti cial Intelligence 146 (2025) 110225


predominantly address payment success in terms of channel imbalance,              infrastructure and algorithms, alongside the development of the cryp-
fee affordability, or node profitability, this work aims to dynamically           tocurrency ecosystem through the introduction of novel computation
optimize pathfinding heuristics in response to network topology and               methods. The second entails the introduction of new architectural solu-
the latent payment failure distribution.                                          tions, which may encompass a range of approaches, from incremental
   This paper makes the following contributions to the field:                     improvements to consensus algorithms to the establishment of novel
                                                                                  cryptocurrency networks and the development of innovative network
    • An extended review of the payment pathfinding solutions and                 designs, as well as the exploration of new architectural programming
      algorithms in the area of PCN is performed, and it is demon-                patterns.
      strated that the problem of payment success failures remains                    Despite the extensive adoption of cryptocurrencies, their value re-
      understudied.                                                               mains highly volatile, which has a significant impact on investment and
    • A comparison of the native LN pathfinding algorithms is pre-                trading decisions. This has led to extensive research into the prediction
      sented, demonstrating that their payment success varies under               of their price. In the first direction of development, recent papers have
      dynamic conditions and is relatively worse when confronted with             presented improved variants of the sine-cosine algorithm and meta-
      a hidden probability distribution of payment failures.                      heuristics for accurate Bitcoin price prediction (e.g., Salb et al., 2022;
    • A reinforcement learning approach is developed for the dynamic              Mizdrakovic et al., 2024). These include the use of recent advances in
      optimization of LN payment pathfinding heuristics. This includes            neural networks (e.g., Milicevic et al., 2023; Nayak et al., 2022). In the
      the creation of a training environment, a RL-agent prototype, and           second direction, price volatility has been managed primarily through
      the design of an efficient reward function.                                 the introduction of projects such as Stablecoin, including Tether (USDT)
    • The proposed approach is validated using a real network snap-               and USD Coin (USDC), which have gained traction as payment solutions
      shot, and the performance of the trained agent is evaluated in a            due to their higher price stability.
      network where the probability of payment failure is stochastically              In view of the limited efficiency of the existing cryptocurrency
      distributed but dependent on the geographical location of nodes.            infrastructure, which is affected by network congestion and low trans-
      The proposed approach demonstrates superior performance in                  action speeds, a number of new architectural solutions have been
      terms of payment success while maintaining balance across other             proposed. The most significant of these are changes to consensus al-
      key metrics in comparison to the existing methods.                          gorithms, which represent a core technology of the blockchain. A
    The remainder of this paper is organized as follows. The following            number of highly effective consensus algorithms, including Proof of
section presents an in-depth examination of the current developments              Stake, Delegated Proof of Stake and others, have been proposed (for
in global payment blockchain-based and channel-based networks, iden-              a review, see Xiong et al., 2022; Hussein et al., 2023). The selection
tifying research gaps and defining the focus of this work. The Methods            of a consensus algorithm entails a number of trade-offs, with the
section presents the methodology that has been adapted for this study,            choice of an algorithm potentially having significant implications for
together with a detailed description of the proposed solution, the re-            the performance and stability of a blockchain network. Furthermore,
quired data preparation procedures, and the chosen evaluation strategy.           given that the architectural characteristics of a blockchain inform
The Results section presents the findings of the experiments, accompa-            significant design decisions, a number of architectural patterns have
nied by the relevant figures. This is followed by a discussion of the             been identified for blockchain application software architectures. These
conclusions that can be drawn from the results and suggestions for                are intended to address the unique challenges presented by blockchain
future work.                                                                      technology. For further details, please see, for example, Alzhrani et al.
                                                                                  (2023) and Xu et al. (2017).
2. Literature review and research focus                                               A contrasting perspective is offered by the second direction of
                                                                                  development, which concerns the design of ‘‘layer-two’’ networks con-
2.1. Current development of global payment blockchain networks                    structed on top of existing (layer-one) cryptocurrency infrastructures.
                                                                                  Examples of such infrastructures include the Lightning Network (Poon
    The advent of global payment networks utilizing blockchain-based              and Dryja, 2016) for Bitcoin and the Raiden Network (Raiden, 2024)
cryptocurrency has profoundly transformed the global financial land-              for Ethereum. In contrast with the aforementioned solutions, layer-two
scape, offering expedited, cost-effective and decentralized alternatives          protocols facilitate the scaling of blockchains without modifying the
to traditional banking systems. Cryptocurrencies, constructed upon                underlying layer-one trust assumptions and do not extend or replace
blockchain technology, facilitate peer-to-peer transactions without the           the existing consensus mechanisms (Gudgeon et al., 2020).
involvement of intermediary institutions, thereby enabling cross-border               Layer-two protocols facilitate the execution of transactions, desig-
payments with reduced fees and delays. Nevertheless, a number of                  nated as off-chain through the utilization of private and authenticated
significant obstacles impede the comprehensive integration of cryp-               communication channels, as opposed to broadcasting each transaction
tocurrencies into the realm of conventional monetary transactions and             on the parent blockchain. The advent of layer-two networks and a
the broader financial system. These include concerns pertaining to price          novel category of blockchain-related payment networks, designated as
volatility, scalability challenges, security vulnerabilities, and environ-        PCNs, has resulted in a notable decline in the transaction load on
mental impact.                                                                    the underlying blockchains. However, there is potential for further
    Following the advent of Bitcoin, the inaugural cryptocurrency to              enhancements in aspects such as privacy and network performance
exemplify the potential of decentralized payments, in 2009, it was ob-            optimization, as evidenced by Dasaklis and Malamas (2023).
served that transaction speeds were relatively slow and fees were high
during periods of network congestion. This resulted in limitations to             2.2. Current development of global payment channel networks
its effectiveness as a global payment network. The advent of Ethereum
also constituted a notable advancement in the evolution of global                     The fundamental premise of PCNs and LN, as a case in point, is to
payment networks. However, it was still perceived to exhibit inherent             facilitate the majority of payment transactions outside the blockchain,
scalability constraints, vulnerabilities pertaining to smart contracts, and       thereby enabling nodes to establish two-way (bidirectional) payment
centralization concerns, as evidenced by the findings of Mehar et al.             channels (Wikipedia, 2024). Each payment channel functions as a joint
(2019).                                                                           account for the respective channel members. To guarantee the security
    In order to address these challenges, developers and researchers              of the system, the opening of a payment channel necessitates the
have pursued two principal avenues of inquiry. The first of these                 transfer of funds to a shared address on the blockchain. Such funds
involves the proposal of technological enhancements to the existing               serve as a secure repository for potential multiple off-chain transactions

                                                                              2
D. Valko and D. Kudenko                                                                              Engineering Applications of Arti cial Intelligence 146 (2025) 110225


between the two parties. Upon the closure of a payment channel, the               applications. These include micropayments, cross-border and remit-
final balance is settled on the underlying blockchain (Avarikioti et al.,         tance payments, IoT and machine-to-machine payments, decentralized
2023).                                                                            exchanges, transfers of digital assets and LN-related services (Dasaklis
    This approach circumvents the dissemination of each blockchain                and Malamas, 2023). As of 28 May 2024, the LN comprises 51,769
transaction to the entire network. Conversely, they employ the costly             observed nodes and 51,925 payment channels (see Fig. 1), with an
and low-yield blockchain solely as a means of resolving disputes.                 estimated network capacity of 4851.75 BTC (Stat1ml, 2024). The con-
PCNs promise to conduct off-chain transactions in seconds, rather than            siderable success of the LN has guaranteed its position as the foremost
minutes or hours, while maintaining the security of assets, reducing              PCN implementation (Camilo et al., 2022).
fees and improving the scalability of the blockchain (Gudgeon et al.,
2020).                                                                            2.2.1. Lightning network topology and dynamics
    It is also noteworthy that each node in such a network is capable of              The preeminent position of the LN necessitates a comprehensive
establishing payment channels with numerous other nodes. This results             examination of its growth and topological characteristics (see, for
in the formation of a network that enables the transfer of funds to               example, Camilo et al., 2022; Valko and Kudenko, 2024; Seres et al.,
neighboring entities via the intermediary of other peers. The establish-          2020), including the formulation of novel topological improvements
ment and maintenance of a payment channel necessitates the blocking               (see, for example, Mahdizadeh et al., 2023). Furthermore, numerous
of funds, thereby generating opportunity costs for the intermediary. In           studies have also identified shortcomings in the current structure of
order to reduce these costs, intermediary nodes receive a commission              the network, particularly in terms of its vulnerability to various attacks,
(fee) for their services.                                                         including those of a topological nature (see, for example, Rohrer et al.,
    The mechanics underlying PCNs have attracted considerable at-                 2019). It is therefore crucial to gain an understanding of the topology
tention in recent literature (for a review, see Gudgeon et al., 2020).            and dynamics of the network in order to ascertain the viability of
Furthermore, the cryptographic foundations of PCN security proofs                 optimization based on the identified patterns and related heuristics.
have been subjected to analysis (Malavolta et al., 2017; Beres et al.,                The LN is an evolving network, its growth being contingent upon the
2020; Kappos et al., 2021). Additionally, efforts have been made to               addition of new users to the network and the establishment of payment
comprehend the game-theoretic aspects of these networks from two                  channels with existing nodes. The considerable scale of this network is
distinct perspectives: security (Hüttel and Staroveski, 2020; Avarikioti          illustrated in the figure below (see Fig. 1).
et al., 2020) and economic (Avarikioti et al., 2023; Van Engelshoven                  In the most recent analytical paper (Mahdizadeh et al., 2023), the
and Roos, 2021).                                                                  average degree of nodes in the LN was found to be 8.46, with a high
    In addition to the LN, which will be discussed in detail in the follow-       degree of skewness, and the density was determined to be 0.00046.
ing section, a number of PCNs have been developed as continuations                This indicates that the network is relatively sparse. The effective di-
of the LN concept for other blockchain networks (for a detailed review,           ameter is approximately 12, and the average shortest path length in
see Khojasteh and Tabatabaei, 2021). The most notable of these are:               the network is 3.65. Although this calculated value is independent
    – The Raiden Network (Raiden, 2024) adapts the LN concept for                 of the payment channel capacities, it indicates that nodes are highly
Ethereum, thereby reducing transaction costs by up to seven times com-            accessible and that multi-hop payments with excessive hops are absent
pared to native Ethereum. Raiden is an efficient system for processing            in the network (Mahdizadeh et al., 2023).
micropayments and supports off-chain transactions with any token that                 The number of bridges is approximately 8560 payment channels.
adheres to the standard Ethereum token API.                                       As a consequence of the existence of a considerable number of nodes
    – The Sprites Network (Miller et al., 2019) is an Ethereum-focused            with a degree of 1, a significant proportion of the calculated bridges
network inspired by the LN and Raiden, but with the objective of                  are edges associated with single-degree nodes. The number of bridges
reducing collateral costs in off-chain payments. In contrast to Lightning         in the network, excluding those associated with single-degree nodes,
and Raiden, where the cost of collateral increases in line with the length        is 174. This suggests the existence of a substantial number of crucial
of the payment chain, Sprites maintains a constant cost by leveraging             payment channels for maintaining network connectivity (Mahdizadeh
Ethereum’s smart contracts.                                                       et al., 2023).
    – Ripple (Schwartz et al., 2014) is a payment settlement system and               The local clustering coefficient of a node is an indicator of the level
cryptocurrency exchange network that is capable of processing trans-              of connectivity among its neighboring nodes. This value indicates the
actions on a global scale. It does not employ blockchain technology to            degree of similarity between the sub-graph formed by a node’s neigh-
achieve consensus, instead it utilizes a hash tree to summarize data and          bors and a clique comprising those neighboring nodes. The mean value
compare this across its validating servers.                                       of local clustering coefficients for network nodes is 0.1231. This value,
    – Duplex Micropayment Channels (Decker and Wattenhofer, 2015)                 excluding zero values, would be 0.3255 (Mahdizadeh et al., 2023). The
employs Bitcoin’s timelock functionality (Roche, 2019) to facilitate              assortativity coefficient of degrees in a network indicates the tendency
routed payments with minimal delay.                                               of nodes to form connections with other nodes that have similar de-
    – Teechan (Lind et al., 2019) is a full-duplex payment channel                grees. This characteristic is measured on a scale between 1 and −1,
framework on Bitcoin for micropayments. The system employs multi-                 where a positive value indicates a tendency for high-degree nodes to
signature time-locked transactions, thereby enabling concurrent pay-              connect with other high-degree nodes, and a negative value indicates
ments in both directions.                                                         a tendency for high-degree nodes to connect with low-degree nodes. In
    – Plasma (Poon, 2017) is a framework for the execution of smart               the LN, the assortativity coefficient of degrees is −0.195, as indicated
contracts on the Ethereum platform, utilizing a hierarchical struc-               in Mahdizadeh et al. (2023). This value reflects the interconnection of
ture of blockchains. Child chains are secured by parent chains, and               low-degree nodes with high-degree nodes.
computation is framed using MapReduce functions.                                      In accordance with the principles of network dynamics, the LN is
    There is sufficient empirical evidence to suggest that the most               observed to incorporate, on average, 4%–5% of novel nodes and 10%–
common PCNs, such as LN, Reiden, and Ripple, exhibit similarities in              15% of novel payment channels on a monthly basis. It is notable that
their distribution, topology, and dynamics (Rohrer et al., 2019; Seres            the majority of these channels serve to supplant existing ones (Valko
et al., 2020; Rebello et al., 2022; Chen et al., 2022a). Consequently,            and Kudenko, 2023). It was found that almost 50% of nodes could be
these networks can be approximated by scale-free networks (Seres et al.,          theoretically reached by two payment hops, 93% by three hops, and
2020).                                                                            almost 98% by no more than five hops (Shell, 2022; Beres et al., 2020).
    As evidenced by the relevant literature, the LN is the most im-               This also represents an advantage in terms of dynamic stability, which
pactful PCN, with numerous current and potential business-related                 can be utilized by intelligent agents or other approximation algorithms

                                                                              3
D. Valko and D. Kudenko                                                                                     Engineering Applications of Arti cial Intelligence 146 (2025) 110225




Fig. 1. Lightning Network graph visualization.
Note. The Lightning Network graph visualization of nodes and payment channels, as of 23 May 2024 (LNRouter, 2024). For the sake of clarity, only nodes with more than 15
payment channels are shown.


for performance optimization purposes.                                                     • Insufficient funds: A common cause of payment failure is when
   It can thus be concluded from these topological characteristics                           the sender lacks adequate outbound capacity to cover the pay-
that the LN exhibits certain patterns that can be approximated using                         ment amount. In PCNs, only the sender’s outbound balance is
admissible heuristics to perform some payment pathfinding algorithms                         spendable (Waugh and Holz, 2020; Pickhardt and Richter, 2021).
with greater efficiency. This conclusion is consistent with the results of                 • Inadequate payment channel connectivity: If the sender’s pay-
other recent studies (Valko and Kudenko, 2024).                                              ment channels do not connect to the recipient, the payment can-
                                                                                             not be routed successfully (D’Angelo et al., 2016; Pickhardt and
                                                                                             Richter, 2021; Davis and Harrison, 2022). Maintaining connec-
2.2.2. Lightning network payment success challenges and the focus of this                    tions with well-established peers – characterized by high uptime,
work                                                                                         substantial capacity, and numerous channels – can enhance the
    A common criticism of PCNs and, in particular, the LN is that                            likelihood of successful payments.
payments do not always succeed (Dasaklis and Malamas, 2023). As doc-                       • Large payment sizes: Larger payments are more prone to failure
umented in River’s research report (River, 2023), the average payment                        due to the increased difficulty in finding a single payment path
success rate on their LN infrastructure ranged from 96.5% to 99.6% (as                       with sufficient liquidity (Waugh and Holz, 2020; Pickhardt and
                                                                                             Richter, 2021).
a maximum in 2023). Other sources have indicated a success rate of
                                                                                           • Channel depletion: Over time, payment channels may become un-
approximately 50% (Stadelmann, 2023). This is a challenging problem
                                                                                             balanced or depleted, especially if used predominantly in one di-
to fully resolve, particularly when developers have limited insight
                                                                                             rection. This imbalance can lead to payment failures (Podiatchev
into the underlying causes of payment failures due to the intrinsic
                                                                                             et al., 2024; Rohrer et al., 2019; Pickhardt and Richter, 2021).
architecture of the network.
                                                                                           • Network attacks: Malicious actors can intentionally cause pay-
   Generally speaking, payment failures in PCN can occur due to                              ment failures through various attacks. Implementing robust secu-
several factors, the most significant are:                                                   rity measures and staying informed about potential vulnerabilities

                                                                                   4
D. Valko and D. Kudenko                                                                                   Engineering Applications of Arti cial Intelligence 146 (2025) 110225


      are crucial for network integrity (Khalil et al., 2023; Weintraub                2024), c-Lightning (CLN) (CLN, 2024) and Eclair (ECL) (ECL, 2024).
      et al., 2021; Rohrer et al., 2019; Kappos et al., 2021).                         They are Dijkstra-based and native for LN client software. Such algo-
    • Pathfinding and routing1 algorithm limitations: The efficiency of                rithms utilizing source-routing have also been the subject of criticism
      the subsequent pathfinding and routing algorithms plays a signif-                due to their limited scalability in large networks (Grunspan and Pérez-
      icant role in payment success (Weintraub et al., 2021; Pickhardt                 Marco, 2019). A number of deterministic algorithms have recently
      and Richter, 2021). Employing more resilient algorithms and                      been developed with the aim of addressing these issues, including:
      routing protocols can mitigate various circumstances such as a                   Flare (Prihodko et al., 2016), SilentWhispers (Malavolta et al., 2016),
      lack of suitable routes, node unreliability, balancing issues etc.               SpeedyMurmurs (Roos et al., 2017), Flash (Wang et al., 2019), ProfitPi-
                                                                                       lot (Camilo et al., 2024) and Fence (Wang et al., 2024). It is regrettable
    Current software implementations of the LN protocol have mainly
                                                                                       that neither of these solutions has been developed into a complete
focused on three strategies for handling mentioned uncertainty of
                                                                                       software package, nor implemented in LN software clients. This is due
sending a payment (Pickhardt and Richter, 2021): (i) incentivizing the
                                                                                       to the necessity for extended online testing and the potential need for
pathfinding algorithm to favor larger channels, (ii) ad-hoc splitting of
                                                                                       topological or infrastructural changes, which may prove unfeasible.
large payment amounts into smaller ones after failed attempts using a
                                                                                       Furthermore, the most prominent of these have been evaluated on
technique called multi-part payments, (iii) using provenance scores of
                                                                                       relatively small samples of a network (e.g. 128 (Wang et al., 2024),
nodes and channels and other data collected during operation of a node
                                                                                       512 (Camilo et al., 2024), 1870 (Wang et al., 2019), 1900 (Chen et al.,
to estimate which nodes and channels might be reliable.
                                                                                       2022b) nodes). This is insufficient to prove superiority over native
    These strategies are not a panacea, because of naive design choices
                                                                                       algorithms in the real world. Furthermore, it has been asserted that
in three main areas (Khojasteh and Tabatabaei, 2021; Sivaraman et al.,
                                                                                       none of the devised algorithms in PCNs are capable of meeting the
2020): (i) payment pathfinding and routing, (ii) timing of payment
                                                                                       desired criteria of scalability, efficiency and effectiveness under real
transaction submission and its final success, (iii) network deadlocks.
                                                                                       dynamic conditions (Gudgeon et al., 2020; Chen et al., 2022b).
Firstly, PCNs such as the Lightning and Raiden Network utilize source-
                                                                                           It is therefore of great importance to address the aforementioned
routing, whereby clients typically select the shortest path from source
                                                                                       issues and to achieve a higher level of payment success in the current
to destination. This approach has the potential to degrade throughput
                                                                                       development of PCNs, particularly in the context of the LN. It is of
for two reasons. First, shortest-path routing can result in congestion in
                                                                                       particular importance to pursue this objective through the utilization of
some payment channels, while others remain underutilized. Second, the
                                                                                       architecturally less expensive methodologies, such as the modification
dominance of funds flowing in one direction can lead to imbalanced
                                                                                       of pathfinding heuristics (Valko and Kudenko, 2024). In the case of
payment channels, with the majority of funds concentrated on one
                                                                                       the LN, it is crucial to develop heuristics that are robust and take into
side (Sivaraman et al., 2020).
                                                                                       account the network topology, as well as the uneven utilization of its in-
    Secondly, the majority of PCNs operate with circuit switching,
                                                                                       frastructure, during the pathfinding phase (Valko and Kudenko, 2024).
whereby transactions are processed immediately and independently
                                                                                       It is therefore the objective of this work to conduct further research
upon arrival. This results in a number of issues, including exceeding
                                                                                       into the development of effective payment pathfinding heuristics, with
the available balance on the selected payment path and imbalance
                                                                                       a view to improving the payment success rates.
problems for large transactions (Khojasteh and Tabatabaei, 2021).
                                                                                           Given the inherently dynamic nature of payment success in LN
    Thirdly, in certain instances, concurrent transactions may interfere               infrastructure, with success rates depending not on financial metrics
with one another, resulting in the network becoming unresponsive                       alone but also on technical characteristics such as transport layer net-
despite the availability of sufficient endpoint balance.                               work performance and node and payment path reliability (River, 2023),
    Addressing these issues involves ensuring sufficient liquidity, es-                this paper presents and evaluates a novel approach to dynamically
tablishing connections with reliable peers, utilizing payment-splitting                optimize pathfinding heuristics using reinforcement learning.
techniques for large transactions, regular or pro-active rebalancing
channels, implementing strong security practices, and adopting effi-                   2.3. Reinforcement learning for the pathfinding task in payment channel
cient routing algorithms. So far, none of the existing solutions have                  networks
been practically implemented to be an improvement over the LN native
algorithms.                                                                                The application of the RL-based approach to address pathfinding
    In summary, PCNs are designed to facilitate high transaction                       tasks in diverse network configurations has been extensively explored
throughput. It is crucial for routers (intermediate nodes) to economi-                 in the literature (for a comprehensive review, see Mammeri, 2019). For
cally balance the opportunity costs in payment channels and encourage                  example, Boyan and Littman (1993) pioneered the integration of the Q-
end-users to accept transactions by ensuring an adequate quality of                    learning algorithm with packet routing, enabling the dynamic learning
payment services. A transaction will only be successful if all payment                 of routing scenarios and the identification of the shortest path within
channels along its path have sufficient funds and all nodes involved                   a network. Gao et al. (2017) put forth a multi-agent routing algorithm
in the transaction are accessible in terms of network connectivity                     that employs Q-learning and backpressure. This approach requires each
and transport layer availability. It is therefore evident that routing                 routing node to possess only local information about its neighboring
protocols and payment pathfinding algorithms, which are employed to                    nodes, thereby enabling the effective resolution of the aforementioned
identify an optimal path for a payment transaction, are of paramount                   task. Mayadunna et al. (2017) and Yang et al. (2019) put forth a
importance.                                                                            RL-based approach to detect malicious routing nodes in mobile ad
    The current pathfinding methods employed in the LN for routing                     hoc networks and wireless sensor networks, respectively. Nevertheless,
payments are discussed in Kumble and Roos (2021), namely LND (LND,                     these developments have been largely confined to specific network
                                                                                       types such as wired networks, wireless networks, ad hoc networks etc.,
   1
                                                                                       and certain sub-types of peer-to-peer networks like IoT blockchain-
      The pathfinding phase involves identifying an optimal route for a payment
                                                                                       based networks, and have only recently begun to be applied to PCNs.
transaction based on network topology and constraints like fees and liquidity.
                                                                                       This specificity means that the definitions of state and action spaces, as
It is a computational process that occurs before payment execution. In contrast,
the routing phase is the real-time execution of the payment along the chosen           well as the design of features, reward functions, and even software im-
route, ensuring funds are transferred while adapting to dynamic network                plementations, are typically tailored to the unique topological and even
conditions. In this work, we focus on the pathfinding solution; however, our           physical characteristics of a certain type of network. This implies that
evaluation also simulates the routing phase, as these two phases are dependent         these RL-based solutions cannot be considered universally applicable or
and influence the final payment success.                                               proven across different network categories.

                                                                                   5
D. Valko and D. Kudenko                                                                              Engineering Applications of Arti cial Intelligence 146 (2025) 110225


    In recent years, numerous attempts have been made to address the             metrics. However, it remains slightly less efficient in terms of pathfind-
dynamics of PCNs using a range of RL-based techniques and solutions              ing execution time, as deterministic algorithms are still part of the
(see Table 1). For instance, Kadry and Gadallah (2021) introduced a              pipeline.
modified RL-algorithm focusing on payment path selection, achieving                  In contrast to the mentioned works, this paper seeks to address the
improved routing success by excluding inactive nodes and updating                issue of payment success by considering the underlying distribution
Q-values for paths as nodes are removed. Luo and Li (2022) used multi-           of payment failures within a network and dynamic balancing across
agent deep Q-learning approach (DQN) to prioritize transactions based            the set of sustainability metrics. This is done by performing a com-
on historical success and fee data, enhancing network throughput, al-            prehensive comparison with native LN pathfinding algorithms using a
though pathfinding task was handled by existing algorithms. Chen et al.          real-world snapshot of the LN.
(2022a) developed PLAC, an actor-critic model optimizing throughput
by managing router node transactions, yielding a throughput increase             3. Methods
of up to 34.9% and outperforming Flash algorithm. Davis and Har-
rison (2022) proposed an RL-model for improving node connectivity                3.1. Reinforcement learning preliminaries and the design of the proposed
in small PCN networks. Chen et al. (2024) proposed DRL-PCR, a deep               solution
RL approach to rebalance payment channel capacity, achieving about
40% payment success with the average LN configuration but without
                                                                                      The RL-based algorithm considers a general situation in which an
addressing the network dynamics. Many recent studies have also aimed
                                                                                 agent acts as a decision maker and seeks to make decisions in an inter-
to increase the profitability of the node or security of payments without
                                                                                 active environment. At each decision point, the environment transmits
addressing the pathfinding task itself (e.g., Asgari et al., 2022; Papadis
                                                                                 the current state to the agent. In response, the agent executes an action
and Tassiulas, 2023; Song et al., 2024; Qiao et al., 2024).
                                                                                 in accordance with its policy. Subsequently, the environment returns
    The most proximate to the present work is the recent introduction
                                                                                 a reward, which serves as an evaluation of the efficacy of the action.
of a method designated as one-step path prediction by Valko and
                                                                                 Furthermore, the environment progresses to a new state in accordance
Kudenko (2023). They presented an RL-based solution for a network
                                                                                 with the action executed. Such a setting should satisfy the requirements
agent that learns its neighborhood and utilizes local knowledge to
                                                                                 of a Markov decision process (MDP). A MDP is defined as a tuple,
address the payment pathfinding problem. Although this preliminary
                                                                                 𝑀 = (𝑆 , 𝐴,  , 𝑅, 𝛾), where 𝑆 and 𝐴 are the state and action space,
solution outperforms native algorithms in terms of execution time by
                                                                                 respectively. The transition function, denoted by  (𝑠, 𝑎, 𝑠′ ) ∶ 𝑆 ×𝐴×𝑆 →
reducing computational cost at each pathfinding step, it was designed
                                                                                 [0, 1], is a probability distribution over the set of states, which describes
and tested on a relatively small sample of the network snapshot (up
                                                                                 the likelihood of reaching a state 𝑠′ from a given state 𝑠 after executing
to 100 nodes). Furthermore, the payment success metric, defined as
                                                                                 an action 𝑎. The reward function, denoted by 𝑅(𝑠, 𝑎, 𝑠′ ) ∶ 𝑆 ×𝐴×𝑆 → 𝑅,
the number of transactions successfully completed by an agent within
                                                                                 assigns a numerical reward to a state transition from 𝑠 to 𝑠′ with respect
its own neighborhood, could only be evaluated to assess the agent’s
                                                                                 to the executed action 𝑎. A policy, defined as a function 𝜋(𝑠, 𝑎) ∶ 𝑆×𝐴 →
capacity to adapt to changes in network topology. In the absence of
                                                                                 [0, 1], specifies the manner in which the agent should act within the
substantial changes in the neighborhood, the agent achieves a payment
                                                                                 environment, through the establishment of a probability distribution
success rate of approximately 96%. However, the solution lacks scala-
                                                                                 over all actions in every state. The decay factor, defined as 𝛾 ≤ 1, is
bility and cannot be deployed in a real-size network where transactions
are conducted sequentially across the entire graph. Consequently, a fair         employed to ascertain the anticipated discounted return and the value
comparison cannot be made with the current work.                                 function, designated as 𝜈(𝑠). In light of the aforementioned, the agent
    It can be concluded that the infrastructural roots of payment failures       is tasked with learning the policy 𝜋(𝑠, 𝑎) that maximizes the expected
remain uncovered, given that existing RL-based solutions mainly con-             discounted return. It is established that the optimal policy is that which
sider the payment success as a result of payment channel imbalance               exhibits the maximum expected discounted return (Sutton and Barto,
or fee affordability and routing profitability. Furthermore, none of             2018).
the aforementioned works has considered the payment pathfinding                       An off-chain PCN, in particular the LN, can be modeled as a
problem itself (with the exception of Valko and Kudenko, 2023) or how            graph, denoted by 𝐺 = (𝑉 , 𝐸), where 𝑉 represents the set of vertices
to obtain candidate payment paths in an effective manner. There is still         (i.e. nodes) and 𝐸 represents the set of edges (i.e. payment channels).
considerable scope for substantial optimization (Chen et al., 2022a). It         To complete a payment demand, the sender initiates the set (one or
should be noted that only work (Valko and Kudenko, 2023) conducted               more) of PCN transactions, denoted by 𝑇 . For each transaction 𝑡 ∈
a comparison of the proposed solution with the native pathfinding                𝑇 , the sender is responsible for determining the candidate payment
algorithms (LND, CLN and ECL) that are currently in use on a full-scale          paths, denoted by 𝑃 , to send those transactions. Each candidate path,
basis.                                                                           represented by the sequence of routing nodes 𝑝 = [𝑢, … , 𝑣], is defined
    It is important to note that most RL-based solutions (see Table 1) can       as a path from the sender node 𝑢 to the receiver node 𝑣, with the
be classified as integrated, meaning the RL-agent operates downstream            condition that 𝑢 ≠ 𝑣. It is the responsibility of each routing node
and makes high-level decisions (such as payment path selection, fee              to select a candidate payment channel, or alternatively a sequence
setting, or hops prioritization) based on the output of a deterministic          of payment channels, to forward the payment. For simplicity, the
pathfinding algorithm. Only two studies propose independent solutions,           predicted payment path 𝑝 can be determined by selecting the cheapest
where the RL-agent makes decisions based solely on the state of the              channels among a sequence of routing nodes [𝑢, … , 𝑣].
network, without relying on deterministic algorithms in its pipeline.                 In the proposed solution, the PCN-graph 𝐺 is treated as the core
This independent approach can be more efficient in terms of computa-             component of the environment, which implements the agent as a
tional complexity and runtime during the payment pathfinding phase.              mapping from states to actions to be taken. It is evident that at each
However, due to the large and complex nature of real-world PCNs, this            stage of the process, the agent receives a transaction 𝑡, together with the
approach often faces scalability challenges, such as computationally             associated source and destination nodes 𝑢, 𝑣. The agent then requests a
expensive state representations and a vast action space.                         state, designated as 𝑠𝑢,𝑣 ∈ 𝑆, which is constructed by the environment
    The approach proposed here is instead a hybrid architecture, where           from a so-called neighborhood observation (Fig. 2). The state is defined
the RL-agent operates upstream, directly influencing the heuristic used          by a vector subset of the features of the nodes adjacent to 𝑢 and 𝑣, with
by a deterministic algorithm during the pathfinding phase. This hybrid           a total of 𝑛 features. From a technical standpoint, the value of 𝑛 can be
approach avoids the scalability issues inherent in fully independent             less than or equal to the number of existing nearest neighbors of 𝑢 and
approaches and enables dynamic optimization across various stochastic            𝑣, multiplied by the number of their observed features.

                                                                             6
D. Valko and D. Kudenko                                                                                      Engineering Applications of Arti cial Intelligence 146 (2025) 110225



Table 1
RL-based techniques and solutions for LN.
 Solution       Task                   Goal              Validation        Tested network     Solution               Reported             Sustainability        Comparison
                                                         network size      model/             architecture           average success      considerations        with native
                                                                           scenario                                  rates in                                   algorithms
                                                                                                                     best/average
                                                                                                                     setup
 MARL           Payment path           Select a          Simulated         The effect of      Integrated             95/80% MARL          –                     –
 (Kadry         selection. The set     payment path      full-sized        increasing the
 and            of potential           with the lowest   network of        number of
 Gadallah,      payment paths is       transaction       80,591 nodes      offline nodes
 2021)          determined by          fees.             and 245,394       and payment
                Breadth-First                            edges.            channel
                Search.                                                    imbalances.

 DRL-PCR        Payment channel        Rebalance         Full-sized        The effect of      Integrated             82/40% DRL-          –                     –
 (Chen          rebalancing            payment           network           scaled channel                            PCR+Flash
 et al.,        request prediction.    channels to       snapshot of       balances and
 2024)          The set of             improve           9021 nodes        the growing
                candidate paths is     transaction       and 128,984       amount of
                defined by the         throughput and    payment           transactions.
                Dejkstra shortest      success ratios.   channels.
                path or Flash
                algorithm.

 LPS (Luo       Priority               Prioritize hops   Small-world       The effect of      Integrated             92/78%               –                     +∕−
 and Li,        assignment             over the          sample of 18      manipulated                               LPS+LND
 2022)          problem. The           payment path      nodes and 36      forwarding
                candidate payment      to maximize       payment           payment
                path is defined by     the transaction   channels.         channel
                LND.                   rate and                            capacity.
                                       minimize the
                                       forwarding
                                       fees.

 PLAC           Defining the           Maximize          Full-sized        The effect of      Integrated             47/7%a               –                     –
 (Chen          maximum amounts        long-term         network           scaled payment                            PLAC+EDJ
 et al.,        to be sent through     throughput.       snapshot, but     channel
 2022a)         the payment                              the size is not   balances and
                channels between                         reported. 40      the number of
                router nodes.                            nodes under       nodes under
                                                         PLAC control.     control.

 RebEL          Developing             Maximize the      Simulated         PCN-network        Integrated             –                    –                     –
 (Papadis       payment channel        node long-        network, the      simulator,
 and            rebalancing policy.    term average      size is not       varying fee
 Tassiulas,                            expected profit   reported.         proportion.
 2023)                                 from fees.

 BiLSTM-        Fee-setting            Achieve the       Simulated         Random             Integrated             –                    –                     –
 PPO            problem.               optimal fee       network of        transactions
 (Song                                 settings and      100-400 nodes.    around one
 et al.,                               revenue for                         simulated
 2024)                                 nodes.                              node.

 RLPath         Payment channel        Accurately        Network           The effects of     Independent            –                    –                     –
 (Qiao          balance prediction,    discover all      snapshots of      graph
 et al.,        subsequent             payment           513–3732          partitioning
 2024)          pathfinding.           channel           nodes and         methods and
                                       balances.         8474–55,428       parallelization.
                                                         balances.

 DyFEn          Maximize income        Define possible   Simulated         Simulated          Integrated             –                    –                     –
 (Asgari        for the node.          fees for          network of        transactions
 et al.,                               payment           100–500           across 3
 2022)                                 channels of the   nodes.            random nodes.
                                       node.

 MBI-A2C        Connection point       Propose nodes     Simulated         Scenarios with     Integrated             –                    –                     –
 (Davis         recommendation.        to open           network of 128    different
 and                                   payment           nodes.            openings
 Harrison,                             channels with,                      budget.
 2022)                                 to maximize
                                       the node
                                       expected
                                       routing
                                       opportunities.

                                                                                                                                                       (continued on next page)




                                                                                  7
D. Valko and D. Kudenko                                                                                           Engineering Applications of Arti cial Intelligence 146 (2025) 110225



Table 1 (continued).
    RLA           Payment                Find an             Sampled            Random              Independent           96/78%               +                     +
    (Valko        pathfinding.           acceptable          sub-networks       transactions in
    and                                  payment path        of 50–100          sub-networks
    Kudenko,                             with less           nodes.             of different
    2023)                                runtime.                               sizes.

    Current       Payment                Find payment        Full-sized         Varied the          Hybrid                100/66%              +                     +
    work          pathfinding.           paths that are      network            average
                                         expected to be      snapshot of        probability of
                                         successful with     9890 channels      payment
                                         balanced            and 5966           channel failure
                                         sustainability      nodes.             within a range
                                         metrics.                               of 1% to 6%.
a
    Normalized throughput instead of payment success rates is reported.




                                                   Fig. 2. The proposed agent-based design of the network pathfinding engine.


    In principle, based on the observed features of neighboring nodes                          Given that, the current state representation 𝑠𝑢,𝑣 is a matrix con-
and payment channels for a given transaction 𝑡, as well as the in-                         taining the characteristics of payment channels, derived from up to 𝑘
formation about subsequent successful payments, the agent is able                          available payment channels established between up to 𝑚 neighboring
to discern the hidden dynamic and topological characteristics of the                       nodes of source 𝑢 and destination 𝑣 nodes, see Fig. 3.
given neighborhood sub-networks. This local knowledge will permit                              To determine the set of neighboring nodes, we use standard ego-
the agent to further modify the pathfinding heuristics that have been                      graph sampling2 with a radius of 𝑚. Given that the effective LN di-
previously employed in order to dynamically predict and improve                            ameter does not exceed 8 hops, on average, and after testing different
payment success. Accordingly, in the proposed implementation, the                          𝑘-values to achieve reasonable learning speed, we found an efficient
agent’s selected action, represented by 𝑎𝑢,𝑣 ∈ 𝐴, returns the value of                     configuration with 𝑘 = 15 and 𝑚 = 5. This combination allows the
the modifier, denoted by 𝜂, 𝑎𝑢,𝑣 = 𝜂(𝑢, 𝑣). This value will subsequently                   agent to develop admissible pathfinding heuristics using limited and
modify the pathfinding heuristic, represented by ℎ, at the next payment                    dynamic nearest neighbors information, which is a common situation
pathfinding step (see Fig. 2).                                                             in real-world payment networks.
                                                                                               In order to maintain consistency with the preliminary results ob-
    As shown in Fig. 2, this paper presents a hybrid architecture based
                                                                                           tained earlier using static multi-criteria optimization (for further de-
on reinforcement learning that combines a trained agent and a native
                                                                                           tails, please refer to Valko and Kudenko, 2024), the reward function
deterministic pathfinding algorithm. This novel hybrid approach uti-
                                                                                           𝑟 is defined as the weighted payment success of the chosen action, as
lizes the fast and complete pathfinding solutions provided by Dijkstra-
                                                                                           outlined in Eq. (1).
based deterministic algorithms, while adapting to the dynamic and
probabilistic nature of payments in the network.                                                      𝑠𝑢𝑐 𝑐 𝑒𝑠𝑠([𝑢, 𝑣]) ⋅ 𝑑 𝑖𝑠𝑡([𝑢, 𝑣]) ⋅ 103
                                                                                           𝑟(𝑎𝑢,𝑣 ) =            ∑                            ,                 (1)
    In light of the aforementioned definitions and the availability of                                              𝑖𝑛𝑡CO2 [𝑢, 𝑣]
both observable payment channel characteristics (payment delay, fee                        where 𝑠𝑢𝑐 𝑐 𝑒𝑠𝑠([𝑢, 𝑣]) is equal to 1 if the payment was successful, 0
ratio) and sustainability-related payment path metrics (estimated geo-                     otherwise; 𝑑 𝑖𝑠𝑡([𝑢, 𝑣]) = |𝑝𝑢,𝑣 | is the payment path length; 𝑖𝑛𝑡CO2 is the
graphical distance, carbon intensity, number of inter-country and inter-
continental hops) as an implemented state representation 𝑠𝑢,𝑣 , it can be
posited that a well-trained agent should be capable of outperforming                         2
                                                                                               Ego-graph sampling returns induced sub-graph of neighbors centered at a
native static heuristics in such a dynamic environment.                                    given node within a given radius.


                                                                                       8
D. Valko and D. Kudenko                                                                            Engineering Applications of Arti cial Intelligence 146 (2025) 110225




                                                           Fig. 3. The PCN state representation.




average carbon intensity of the geographic locations of the involved             training in the original PPO paper (Schulman et al., 2017) were used,
nodes, as defined in Valko and Kudenko (2024).                                   as they are expected to be effective in most cases (see Appendix A).
    In the event of a successful payment within a transaction, the agent
receives a positive reward, the value of which is dependent upon                 3.2. Data and preparation
the selected payment path length and the carbon intensity metric. By
optimizing this reward, the agent is able to discern the optimal value               In this study, LN data snapshots (Decker, 2020) were employed,
of the heuristic modifier, which enables the pathfinding algorithm to            which have been extensively documented in Zabka et al. (2022). Data
identify and dynamically prioritize more efficient payment paths.                collection is based on multiple c-lightning nodes that synchronize their
    The use of average carbon intensity as one of the reward compo-              view of the network topology by exchanging gossip messages. Each node
nents targets the RL-agent to choose payment paths that minimize the             filters out duplicated or outdated messages and updates its internal
corresponding total carbon intensity and excessive number of inter-              view accordingly. To preserve this view across restarts, the node saves
continental and inter-country hops or at least maintain a balance                these messages in a snapshot file and occasionally compresses it by
between acceptable options. This allows payments to be not only                  removing outdated messages (Decker, 2020). Snapshots of the available
successful, but also effective from the sustainability perspective, and          data can thus be used to reconstruct the network topology at a specific
the location based average carbon intensity has already been proven              point in time. This approach has been widely employed to study the
in Valko and Kudenko (2024) as a good metric for this purpose.                   topology and dynamics of PCNs; see, for example, Zabka et al. (2022),
    The proposed solution is novel in that the agent’s actions modify            Pietrzak et al. (2021) and Lin et al. (2020).
the pathfinding heuristics only within a given range using a discovered              Based on the aforementioned data and utilizing the NetworkX
latent payment failure distribution. This differs from our preliminary           python library, a real network topology was reconstructed and av-
work (Valko and Kudenko, 2023), in which the agent replaced the                  eraged around the most recent available timestamps. Nodes lacking
native pathfinding algorithm. As a result, the payment pathfinding               established or deprecated payment channels were excluded, as such
remains transparent and compliant with the approved PCN protocols.               nodes are unable to participate in transactions. Therefore, the final
From a technical standpoint, this solution preserves scalability while           graph comprises 59,890 payment channels and 5966 nodes, with a
maintaining the same level of privacy of the native pathfinding algo-            maximum of 1544 neighboring nodes per node.
rithm. The agent has access only to the dynamic state of a given number              To ensure a fair comparison with baseline approaches, the same
of neighboring nodes that are publicly accessible at any given time.             data preparation pipeline as that described in Valko and Kudenko
    Such pathfinding problems are expected to be effectively addressed           (2024) was reused. The sole distinction pertained to the assignment of a
by Proximal Policy Optimization (PPO), which is a family of policy opti-         randomly generated failure probability to each payment channel within
mization RL-based methods that employs multiple epochs of stochastic             the reconstructed network, based on the topology and geographic
gradient ascent to perform each policy update. These methods exhibit             locations. This was done to model the dynamics of a real payment
the stability and reliability of trust-region methods, yet demonstrate           network, with average payment failure rates ranging from 1% to 6%,
superior overall performance and broader applicability (for further              thus ensuring that the failure probability of inter-continental channels
details, see Schulman et al., 2017). Furthermore, there is evidence that         is twice that of inter-country payment channels. The selected range is
PPO-based agents are capable of solving specific pathfinding tasks and           consistent with pertinent statistics regarding LN failures (see Fig. 4),
of adapting such goal-based heuristics more effectively (Skrynnik et al.,        as well as the statistics of delays in the global network infrastructure
2022; Asgari et al., 2022).                                                      resulting from the routing of TCP packets through inter-continental
    The proposed solution employs the Stable-Baselines3 library, their           internet channels (Azure, 2024; Chaudhry and Yanikomeroglu, 2022).
PPO implementation (Ppo, 2021) and the goal-based learning envi-                 Therefore, a global distribution of payment failures dependent on the
ronment framework (Brockman et al., 2016), extended according to                 network topology and geographical features was obtained, which is in
the aforementioned definitions. The hyperparameters recommended for              close alignment with the real world.

                                                                             9
D. Valko and D. Kudenko                                                                                          Engineering Applications of Arti cial Intelligence 146 (2025) 110225




       Fig. 4. Payment success rates in LN, %. Note. The most recent statistics regarding the payment success rate utilizing River’s LN infrastructure, see River (2023).




    In order to assess the efficacy of the proposed approach, 10,000                       heuristic ℎ introduces a minor cost to the native cost functions in the
transactions were pre-generated in a uniform distribution over the                         event that the payment nodes are situated in disparate countries or con-
network graph, with a logarithmic distribution of payment amounts.                         tinents. Consequently, the corresponding carbon intensity is increased.
These were then divided into a training and test set, with a ratio of 0.7                  The value of ℎ, which provides a gentle incentive for pathfinding
and 0.3, respectively. The training set was then employed to train a RL-                   algorithms to perform location-based searches (Valko and Kudenko,
agent utilizing each of the three native algorithms, namely LND, CLN                       2024), is employed as a baseline approach for this work with Eq. (2).
and ECL. As the algorithms in question employ disparate pathfinding                                            𝑖𝑛𝑡CO2 [𝑣] + 𝑖𝑛𝑡CO2 [𝑢]
heuristics (for further details, see Kumble and Roos, 2021; Kumble                         ℎ(𝑢, 𝑣) = 𝜂(𝑢, 𝑣) ⋅
                                                                                                                          2                                     (2)
et al., 2021), three distinct types of agents were developed, each trained
                                                                                            +(1 − 𝜂(𝑢, 𝑣)) ⋅ (𝑖𝑛𝑡CO2 [𝑣] − 𝑖𝑛𝑡CO2 [𝑢]),
in an environment utilizing one of the native algorithms. A comparison
of the performance of these agents was deemed necessary in order to                        where [𝑢, 𝑣] is a payment channel (or path as a sequence of such
select the most efficient one, given that it is evident that the policies                  channels) that is established from node 𝑢 to node 𝑣; 𝑖𝑛𝑡CO2 is the
learned by them are disparate.                                                             average carbon intensity of involved nodes’ geographic locations; 𝜂 is
                                                                                           the heuristic modifier. The modifier value is typically set in the range
3.3. Evaluation strategy and compared algorithms                                           of [−1..1], as defined in Valko and Kudenko (2024).
                                                                                               Notwithstanding the favorable outcome of the static optimization
    As the review has demonstrated, the extant deterministic algorithms
                                                                                           process, which met the requisite sustainability criteria, there has been
(e.g., Flash) exhibit inferior performance with respect to payment suc-
                                                                                           no testing of the results under dynamic infrastructural payment failure
cess relative to the actual statistics of the LN with the native algorithms
                                                                                           conditions. Furthermore, static optimization techniques are typically
(see River, 2023). Additionally, known RL-based solutions are not de-
                                                                                           computationally expensive, rendering them unsuitable for deployment
signed to address the payment pathfinding problem directly and cannot
be meaningfully compared with native algorithms without substantial                        in a solution that encompasses every payment transaction.
revisions or retraining. It was thus resolved to assess the efficacy of the                    The experimental evaluation strategy entails the execution of native
proposed solution in conjunction with the native algorithms, as well as                    algorithms, namely 𝑛𝑎𝑡𝑖𝑣𝑒𝐿𝑁 𝐷 , 𝑛𝑎𝑡𝑖𝑣𝑒𝐶 𝐿𝑁 , 𝑛𝑎𝑡𝑖𝑣𝑒𝐸 𝐶 𝐿 , and the algo-
with their enhanced version following static optimization, which serves                    rithms with enhanced heuristics (ℎ) after static optimization, namely
as a baseline.                                                                             𝑏𝑎𝑠𝑒𝑙𝑖𝑛𝑒𝐿𝑁 𝐷 , 𝑏𝑎𝑠𝑒𝑙𝑖𝑛𝑒𝐶 𝐿𝑁 , 𝑏𝑎𝑠𝑒𝑙𝑖𝑛𝑒𝐸 𝐶 𝐿 . Since the proposed solution uti-
    As previously stated, there are three principal LN software clients                    lizing dynamic RL-agent-based optimization (denoted as 𝑖𝑚𝑝𝑟𝑜𝑣𝑒𝑑𝐿𝑁 𝐷 ,
with integrated solutions based on Dijkstra’s algorithm: LND, CLN and                      𝑖𝑚𝑝𝑟𝑜𝑣𝑒𝑑𝐶 𝐿𝑁 , 𝑖𝑚𝑝𝑟𝑜𝑣𝑒𝑑𝐸 𝐶 𝐿 ), it is evaluated on a fixed pre-generated set
ECL. These have an open-source python implementation (Kumble and                           of transactions, employing a snapshot of a real network with a hidden
Roos, 2021; Kumble et al., 2021) that has been early adapted to run                        distribution of payment failures (see Section 3.2). In order to obtain
with modified heuristics (Valko and Kudenko, 2024).                                        robust estimates, all of the examined algorithms are to be run five
    In the preliminary work, it was demonstrated that the topology                         times on the same test set of transactions, but with different random
and geographical distribution are significant factors for the LN. Fur-                     seeds due to the presence of stochastic components. Subsequently, the
thermore, the pathfinding heuristics of the native algorithms can be                       payment success rates, defined as the number of successful payments
adaptively modified to achieve sustainability goals. For this purpose,                     divided by the total number of payments, are averaged across all
the modified heuristic and the specific optimal values were proposed.3
                                                                                           runs and algorithms. These rates are then analyzed against the pre-
In addition to the native LND-, CLN- and ECL-heuristics, this modified
                                                                                           defined probability of payment failure using statistical criteria, fitted
                                                                                           regressions and visual analysis. The two-sided Mann–Whitney U-test
   3
     The special heuristic that minimizes the average carbon intensity for the             with standard levels of significance was used for statistically valid
selected payment channels yielded the following results: 𝜂𝐿𝑁 𝐷 = 0.27876,                  comparison of average values because it is based on ranks and the
𝜂𝐶 𝐿𝑁 = 0.25556, 𝜂𝐸 𝐶 𝐿 = 0.35784 (Valko and Kudenko, 2024).                               median and does not require normality for the data distribution.

                                                                                      10
D. Valko and D. Kudenko                                                                                               Engineering Applications of Arti cial Intelligence 146 (2025) 110225




Fig. 5. Payment success rates comparison.
Note. The figure presents the payment success rates for three approaches: the 𝑛𝑎𝑡𝑖𝑣𝑒 algorithms, the 𝑏𝑎𝑠𝑒𝑙𝑖𝑛𝑒 solution based on static optimization, and RL-agents trained on the
corresponding native algorithms (𝑖𝑚𝑝𝑟𝑜𝑣𝑒𝑑). The experiments were repeated five times using the same transaction test set but with different random seeds, leading to different
channel failure probability distributions. The shaded area beneath the fitted curves represents the 95% confidence intervals.


                 Table 2
                 Payment success rates comparison using statistical criteria.
                                   𝑛𝑎𝑡𝑖𝑣𝑒𝐿𝑁 𝐷   𝑏𝑎𝑠𝑒𝑙𝑖𝑛𝑒𝐿𝑁 𝐷   𝑖𝑚𝑝𝑟𝑜𝑣𝑒𝑑𝐿𝑁 𝐷     𝑛𝑎𝑡𝑖𝑣𝑒𝐶 𝐿𝑁    𝑏𝑎𝑠𝑒𝑙𝑖𝑛𝑒𝐶 𝐿𝑁   𝑖𝑚𝑝𝑟𝑜𝑣𝑒𝑑𝐶 𝐿𝑁   𝑛𝑎𝑡𝑖𝑣𝑒𝐸 𝐶 𝐿   𝑏𝑎𝑠𝑒𝑙𝑖𝑛𝑒𝐸 𝐶 𝐿   𝑖𝑚𝑝𝑟𝑜𝑣𝑒𝑑𝐸 𝐶 𝐿
                   𝑛𝑎𝑡𝑖𝑣𝑒𝐿𝑁 𝐷      0.660
                   𝑏𝑎𝑠𝑒𝑙𝑖𝑛𝑒𝐿𝑁 𝐷    –            0.673
                   𝑖𝑚𝑝𝑟𝑜𝑣𝑒𝑑𝐿𝑁 𝐷    ***          **             0.760
                   𝑛𝑎𝑡𝑖𝑣𝑒𝐶 𝐿𝑁      *            –              –                0.724
                   𝑏𝑎𝑠𝑒𝑙𝑖𝑛𝑒𝐶 𝐿𝑁    *            –              –                –             0.718
                   𝑖𝑚𝑝𝑟𝑜𝑣𝑒𝑑𝐶 𝐿𝑁    **           *              –                –             –              0.753
                   𝑛𝑎𝑡𝑖𝑣𝑒𝐸 𝐶 𝐿     –            –              ***              **            **             ***            0.628
                   𝑏𝑎𝑠𝑒𝑙𝑖𝑛𝑒𝐸 𝐶 𝐿   –            –              **               –             –              *              –             0.667
                   𝑖𝑚𝑝𝑟𝑜𝑣𝑒𝑑𝐸 𝐶 𝐿   –            –              ***              *             –              **             –             –               0.654

                 Note. The figure highlights the statistical significance of differences in the median payment success rates among the 𝑛𝑎𝑡𝑖𝑣𝑒 algorithms, the
                 𝑏𝑎𝑠𝑒𝑙𝑖𝑛𝑒 solution after static optimization, and 𝑅𝐿 − 𝑎𝑔 𝑒𝑛𝑡𝑠 trained on the corresponding native algorithms (𝑖𝑚𝑝𝑟𝑜𝑣𝑒𝑑). The experiments were
                 repeated five times using the same transaction test set but with different random seeds, resulting in different payment channel failure probability
                 distributions.
                  * The Mann–Whitney two-sided 𝑈 -test was used, significance levels: 𝑝 < 0.05.
                  ** The Mann–Whitney two-sided 𝑈 -test was used, significance levels: 𝑝 < 0.01.
                  *** The Mann–Whitney two-sided 𝑈 -test was used, significance levels: 𝑝 < 0.001.


4. Results                                                                                        approach (after static optimization) demonstrates superior estimates,
                                                                                                  albeit by a marginal degree. The results for baselines of CLN, LND, and
4.1. Payment success rates comparison                                                             ECL are 0.718, 0.673, and 0.667, respectively.
                                                                                                      These results demonstrate notable discrepancies between the exist-
    As previously stated, the hypothesis was that a well-trained agent                            ing pathfinding heuristics of the native algorithms and simultaneously
could achieve superior results to those obtained through static multi-                            substantiate the efficacy of static pathfinding optimization. However,
criteria optimization, particularly in the presence of hidden dynamic                             further optimization is possible, as the mean payment success rate
failure probabilities. To test this hypothesis, a test sample of trans-                           decreases markedly as the probability of payment channel failures rises
actions was used to compare the baseline approach with the results                                (Fig. 5).
achieved by each trained RL-agent in terms of payment success rate. To                                The proposed 𝑖𝑚𝑝𝑟𝑜𝑣𝑒𝑑 solutions demonstrate superior performance,
corroborate the dynamic stability of the proposed solution compared to                            yet RL-agents trained on disparate native algorithms exhibit markedly
native algorithms and the baseline approach, the average probability of                           disparate outcomes. The ECL-based RL-agent (𝑛𝑎𝑡𝑖𝑣𝑒𝐸 𝐶 𝐿 ) demonstrated
payment failure was varied within a range of 1% to 6% and randomly                                the poorest performance, with a payment success rate of 0.654. In
distributed across the network (see Methods).                                                     contrast, the CLN-based agent (𝑖𝑚𝑝𝑟𝑜𝑣𝑒𝑑𝐶 𝐿𝑁 ) exhibited a relatively
    As illustrated in Fig. 5 and Table 2, the results demonstrate that                            strong performance, with a payment success rate of 0.753 (see Table 2).
among the 𝑛𝑎𝑡𝑖𝑣𝑒 algorithms, CLN exhibits the highest reliability, with                           The LND-based agent (𝑖𝑚𝑝𝑟𝑜𝑣𝑒𝑑𝐿𝑁 𝐷 ), however, demonstrated the best
a median payment success rate of 0.724 in the selected range of the                               performance. The median success rate within the selected failure prob-
probability of payment failure. This is followed by LND and ECL, with                             ability range is 0.760, which represents a 5% improvement over the
median success rates of 0.660 and 0.628, respectively. The 𝑏𝑎𝑠𝑒𝑙𝑖𝑛𝑒                               best median performance of the native algorithms.

                                                                                             11
D. Valko and D. Kudenko                                                                                            Engineering Applications of Arti cial Intelligence 146 (2025) 110225




Fig. 6. Total network throughput and average payment path length comparison.
Note. The figure presents the total network throughput and average payment path length for three approaches: the 𝑛𝑎𝑡𝑖𝑣𝑒 algorithms, the 𝑏𝑎𝑠𝑒𝑙𝑖𝑛𝑒 solution based on static
optimization, and RL-agents trained on the corresponding native algorithms (𝑖𝑚𝑝𝑟𝑜𝑣𝑒𝑑). The experiments were repeated five times using the same transaction test set but with
different random seeds, resulting in different payment channel failure probability distributions. The shaded area beneath the fitted curves represents the 95% confidence intervals.




    It should be noted that the median values provide a simplified                           perform worse than native LN algorithms in a comparable setup (see,
overview of the data, whereas a closer examination of the regression                         e.g., Wang et al., 2024; Chen et al., 2022a).
curves may offer more insight into the actual differences. The ap-
proximation of the empirical data demonstrates a notable disparity in                        4.2. Multi-criteria comparison
performance among the analyzed solutions in the context of increasing
payment failure probabilities, see Fig. 5. In particular, the agent trained                      In practice, increasing payment success probabilities should be con-
on LND, with a payment failure probability of approximately 1% per                           sidered as a goal together with minimizing fees. Naturally, these goals
payment channel, achieves a success rate of 97%, which is very close                         can be contradictory, since node operators can and will choose fees
to the baseline solution. However, as the failure rate increases, the                        freely. Unfortunately, achieving both of these goals are weakly NP-hard
payment success rate of the proposed solution decreases at a slower                          problems (Pickhardt and Richter, 2021).
rate than that of the competitors. With an average failure rate of more                          The objective of multi-criteria optimization is to achieve a com-
than 5%, it outperforms the other solutions by more than 10%, which                          promise between the general characteristics of payment transactions,
represents a highly competitive result.                                                      such as total payment amounts and fee ratios, while simultaneously
    Table 2 presents a similar comparison, but based on the Mann–                            enhancing more crucial target metrics, including payment path length,
Whitney statistical criterion. In the context of this two-sided version,                     the number of inter-continental and inter-country hops, and average
the null hypothesis is that there is no difference. Overall, while the                       carbon intensity. To this end, a detailed comparison of these metrics
applicability of this criterion can be subject to scrutiny, the outcomes                     was conducted for the proposed solution (Fig. 6, see also Appendix B).
align with the visual analysis. The RL-agent trained on the LND algo-                            The agent was trained using a reward function that employs only
rithm (𝑖𝑚𝑝𝑟𝑜𝑣𝑒𝑑𝐿𝑁 𝐷 ) demonstrates the highest payment success rate or                       the most pertinent metrics, including payment success, the length of the
performs not worse than the other agents in all comparisons. Although                        selected payment path, and the carbon intensity. It would be inaccurate
it performs better numerically in all cases, there is a minor statistical                    to describe this case as a full multi-criteria optimization, given that
difference from the CLN-trained agent that becomes significant when                          other criteria, such as the payment fee ratio, also have significance from
using the one-sided version of the criterion. This can be attributed to                      the user’s perspective. Nevertheless, preliminary research has demon-
the superior performance of the LND algorithm itself in comparison                           strated that although the utilization of multiple multi-directional target
to other native algorithms (for further details, see Kumble and Roos,                        metrics in the design of an agent’s reward is theoretically feasible, in
2021), thus additional dynamic optimization enhances its efficacy.                           practice it impairs the agent’s ability to learn effectively.
    Therefore, the proposed approach for dynamic optimization of                                 In consideration of the most significant user metric – payment
pathfinding heuristics demonstrates satisfactory reliability results in the                  fee ratio and the most significant economic metric – total payment
context of an increasing average payment failure probability. Note that                      throughput, it can be concluded that in all cases, RL-agents demonstrate
a comparison of the results obtained with other algorithms, such as                          performance that is not significantly inferior to that of the baseline
Flash, and with other known RL-based solutions is not performed for                          approach (Fig. 6). Furthermore, with regard to sustainability metrics
reasons discussed in the evaluation strategy section (see Section 3.3).                      (e.g., payment path length, number of inter-continental and inter-
These reasons relate to the fact that state-of-the-art algorithms already                    country hops), they even exhibit superior performance, as evidenced

                                                                                        12
D. Valko and D. Kudenko                                                                              Engineering Applications of Arti cial Intelligence 146 (2025) 110225


in Fig. 6, see also Appendix B. It is anticipated that metrics of low                 Second, the maximum packet length size in the LN is 65569 bytes.4
importance may be sacrificed in order to achieve a higher payment                 Thus, having estimates for the average infrastructure use and worst
success rate. For example, 𝑖𝑚𝑝𝑟𝑜𝑣𝑒𝑑𝐶 𝐿𝑁 has the second-best average               case packet size, assuming for simplicity that one hop requires one
payment success rate but ends up using payment paths with higher fees             maximum length packet, executing one transaction of 5 hops results
(although the 𝑖𝑚𝑝𝑟𝑜𝑣𝑒𝑑 version is still better than 𝑏𝑎𝑠𝑒𝑙𝑖𝑛𝑒). Generally,         in 51.14 kgCO2eq.
the RL-agent trained on the LND algorithm (𝑖𝑚𝑝𝑟𝑜𝑣𝑒𝑑𝐿𝑁 𝐷 ) also per-                   Given that the proposed solution reduces the average path length by
forms better in this comparison, as it achieves higher payment success            ∼3 hops, including ∼1 inter-country/inter-continental hop, the result
rates and shorter path lengths without significantly reducing overall             is a ∼60% reduction in the message transmission footprint or 30.69
throughput.                                                                       kgCO2eq. per average transaction. In addition, using average carbon
    Therefore, although the objective of this paper was not to pursue             intensity as one of the reward components and a key optimization
a full multi-criteria optimization, the proposed solution exhibits favor-         metric minimizes the excess emissions associated with the number of
able comparison results and can be regarded as a compelling proof of              inter-continental/inter-country hops and uneven infrastructure usage,
concept.                                                                          making the potential for carbon footprint reduction apparent.


4.3. Scalability and sustainability considerations                                5. Limitations and reproducibility

                                                                                      The present experimental setup inherits some of the limitations of
    Simulation experiments show that the proposed approach performs
                                                                                  the data used and the methods of their processing, thereby ensuring
effectively on a snapshot of a real full-size network, yielding results
                                                                                  comparability with the previously obtained results.
comparable to native algorithms currently used in client software,
indicating that it is appropriately scalable with respect to network size.            The first limitation of this study and its experimental design is
                                                                                  the dynamic and sparse nature of the LN, which may result in slight
    To provide an approximate assessment of the scalability of the pro-
                                                                                  discrepancies in the numerical results obtained from different snap-
posed hybrid architecture in terms of execution time, the average run-
                                                                                  shots. A second limitation is inherent to the geolocation method, which
time per transaction and per payment hop, including the pathfinding
                                                                                  relies on public IP addresses. A significant number of LN nodes utilize
phase, was compared (see Fig. 7).
                                                                                  onion and hidden addresses. Nevertheless, the visible sub-network is
    Fig. 7 demonstrates that the computational complexity of python
                                                                                  sufficiently large to allow for an analysis of the network’s topological
implementations for algorithms within the hybrid architecture in-
                                                                                  features, as previously demonstrated by Howell et al. (2023) and Valko
creases only slightly (it takes less than a second on average to execute
                                                                                  and Kudenko (2024).A further issue that requires consideration is the
a single transaction). This is primarily due to the relative simplicity of
                                                                                  impact of modifying native pathfinding algorithms on security and
policy-based inference within the trained deep RL-model. Typically, for
                                                                                  privacy of the system (Cai et al., 2020), thus further research is required
a neural network with 𝐿 layers and 𝑃 parameters per layer, the compu-
                                                                                  to investigate potential privacy and security issues in light of known
tational complexity is roughly 𝑂(𝐿⋅𝑃 ) for a forward pass. Consequently,
                                                                                  vulnerabilities in the LN, as identified in the previous studies (Beres
the approximate performance in runtime also indicates an acceptable
                                                                                  et al., 2020; Kappos et al., 2021). However, it is important to note that
technical scalability of the proposed solution. Further improvement in            this is an emerging research area (Malavolta et al., 2017; Kappos et al.,
RL-model inference latency is possible with the model compression                 2021; Kumble et al., 2021). The chosen baseline approach (see Valko
approach, in some cases it can reach more than 50% (McEnroe et al.,               and Kudenko, 2024), which prioritizes localized payment channels, is
2024; Valko and Kudenko, 2023), but this is beyond the scope of this              aligned with the idea (Kumble et al., 2021) that sensitive information
work.                                                                             should not traverse unknown payment channels that are either poorly
    It is difficult to estimate the real carbon footprint for globally            described, lack public metadata, or are significantly remote.
distributed networks that dynamically utilize different client hardware.              Deep reinforcement learning, which we employ in this study, has
According to one source, execution of one LN transaction with 6                   made notable advancements across a range of tasks. However, security
payment hops results in emission of about 0.00992 gCO2eq. Barratt                 and privacy concerns associated with deploying robust RL applications
and Scott (2021), and thus has a significant negative impact on the               remain a topic of ongoing debate (for recent discussions, see Mo et al.,
environment. Of course, the emission dynamics in such a network                   2024). For instance, an adversary may manipulate the environment
depends on the network traffic, the number of transactions at a time,             to influence an agent’s actions, thereby misleading it to behave in an
and the performance of the native pathfinding algorithms.                         anomalous manner.
    To give a raw estimate of the environmental impact of the proposed                At this stage, we utilized a RL-agent to adapt native pathfinding
improvements, the transmission of relevant TCP network messages                   heuristics, whose computation algorithms are already documented in
through the worldwide network infrastructure can also be considered.              open source. This approach is intended to prevent malicious code
First, there is a direct correlation between quality of service (QoS)             injection (for further details, see Kumble et al., 2021; Kumble and
in a TCP communication network and carbon emissions. Specifically,                Roos, 2021). Nevertheless, despite the transparency of the code, the na-
carbon emissions increase as QoS (Habib and Marimuthu, 2013) de-                  tive pathfinding algorithms exhibit some degree of ambiguity, thereby
teriorates. This growth is attributed to the increase in the number of            providing a technical safeguard against the potential reconstruction of
packets generated and their subsequent transmission over an unstable              payment paths. In this context, the RL-agent introduces an additional
network. A study of a modestly sized network of 100 nodes operating               element of uncertainty with regard to the extent of exploitation of the
under varying loads and maintaining a QoS of at least 80% showed                  solution space, which is fixed and can be configured by the agent’s pa-
an average emission rate of 0.156 gCO2eq. per byte (Usman et al.,                 rameters. It is therefore our contention that the deployment of RL-based
2017). This figure is an annual average based on the use of natural               methodologies does not result in a diminution of the aforementioned
gas as the primary source of electricity. However, it is difficult to             security measures compared to the baseline solution. However, this is
get a more accurate estimate due to the use of different network                  a topic that warrants further investigation and discourse.
equipment (Usman et al., 2017). According to reference data, the peak
power consumption of modern Ethernet switches, which mainly per-
form the packet routing function, is approximately 4.42𝐸−09 gCO2eq.                  4
                                                                                       https://github.com/lightning/bolts/blob/master/08-transport.md#
per byte (He, 2019; Vishwanath et al., 2015).                                     lightning-message-specification.


                                                                             13
D. Valko and D. Kudenko                                                                                          Engineering Applications of Arti cial Intelligence 146 (2025) 110225




Fig. 7. Transaction execution time comparison.
Note. The figure shows the average execution time of a successful payment transaction, including the pathfinding phase, for three approaches: the 𝑛𝑎𝑡𝑖𝑣𝑒 algorithms, the 𝑏𝑎𝑠𝑒𝑙𝑖𝑛𝑒
solution, and RL-agents trained on the corresponding native algorithms (𝑖𝑚𝑝𝑟𝑜𝑣𝑒𝑑). The experiments were repeated five times using the same test set of 3000 uniformly distributed
transactions but with different random seeds.




6. Conclusion and outlook                                                                       Through the lens of applying deep reinforcement learning to the
                                                                                            pathfinding task in peer-to-peer PCNs, this paper introduces a novel
    The application of reinforcement learning to solve payment                              hybrid architecture that combines a reinforcement learning agent with
pathfinding problems in different types of networks has been the                            native deterministic pathfinding algorithms. This hybrid approach is
subject of extensive study in the literature. Nevertheless, its deploy-                     scalable enough and outperforms native algorithms and state-of-the-
ment in globally distributed peer-to-peer PCNs, such as the LN, has                         art static optimization methods, delivering enhanced reliability and
predominantly concentrated on profitable solutions, while the issue of                      efficiency.
payment success has remained relatively unresolved, especially in the                           The subsequent stages of this research will concentrate on en-
context of multi-criteria optimization and balancing carbon neutrality
                                                                                            hancing multi-criteria dynamic optimization and incorporating it into
of infrastructure. The objective of this paper is to evaluate, contrast
                                                                                            existing LN software clients, with comprehensive real-world testing, as
and enhance the existing pathfinding algorithms in the LN through
                                                                                            well as further investigating of security and privacy concerns. It would
the utilization of reinforcement learning, with the aim of optimizing
                                                                                            be beneficial for future work in this area to focus on further devel-
payment success and, consequently, enhancing infrastructure sustain-
                                                                                            oping environmentally and topologically sustainable algorithms and
ability. This work makes a contribution to the field by considering
reinforcement learning as a dynamic heuristic improvement approach                          heuristics with the aim of achieving robust, carbon-neutral payment
to achieve robust payment efficiency in a geographically distributed                        infrastructure.
network infrastructure.

                                                                                       14
D. Valko and D. Kudenko                                                                                              Engineering Applications of Arti cial Intelligence 146 (2025) 110225

Table 3
Multi-criteria comparison numerical results.
 Target metrics       𝑛𝑎𝑡𝑖𝑣𝑒𝐿𝑁 𝐷        𝑏𝑎𝑠𝑒𝑙𝑖𝑛𝑒𝐿𝑁 𝐷     𝑖𝑚𝑝𝑟𝑜𝑣𝑒𝑑𝐿𝑁 𝐷      𝑛𝑎𝑡𝑖𝑣𝑒𝐶 𝐿𝑁        𝑏𝑎𝑠𝑒𝑙𝑖𝑛𝑒𝐶 𝐿𝑁       𝑖𝑚𝑝𝑟𝑜𝑣𝑒𝑑𝐶 𝐿𝑁       𝑛𝑎𝑡𝑖𝑣𝑒𝐸 𝐶 𝐿       𝑏𝑎𝑠𝑒𝑙𝑖𝑛𝑒𝐸 𝐶 𝐿      𝑖𝑚𝑝𝑟𝑜𝑣𝑒𝑑𝐸 𝐶 𝐿
 Path length          5.316 ±           4.345 ±          4.363 ±           4.983 ±           4.654 ±            4.975 ±            6.152 ±           4.394 ±            7.963 ±
                      1.321             0.871            0.859             1.149             1.039              1.193              1.506             0.932              7.008
 Geographic           10 941.629 ±      8460.456 ±       6688.297 ±        8732.885 ±        8455.904 ±         8380.934 ±         12 527.681 ±      8463.207 ±         14 895.842 ±
 distance             7870.366          6886.841         6162.966          6873.177          6777.604           6763.812           8726.082          6955.616           19 991.512
 Carbon intensity     1836.134 ±        1366.109 ±       1458.250 ±        1762.521 ±        1581.665 ±         1779.814 ±         2083.946 ±        1308.847 ±         2525.357 ±
                      554.085           397.369          418.626           518.335           476.663            546.738            602.682           378.119            1680.744
 Payment locktime     191.246 ±         211.327 ±        209.613 ±         124.620 ±         131.781 ±          130.787 ±          250.477 ±         218.043 ±          397.016 ±
                      96.911            111.678          115.670           60.742            63.985             62.862             127.799           114.684            389.451
 Payment fee ratio    1.014 ±           1.018 ±          1.018 ±           1.225 ±           1.134 ±            1.129 ±            1.014 ±           1.018 ±            1.026 ±
                      0.266             0.267            0.269             0.803             0.638              0.621              0.266             0.267              0.270
 Number of            1.296 ±           1.006 ±          0.722 ±           1.028 ±           0.998 ±            0.929 ±            1.388 ±           0.986 ±            1.445 ±
 inter-continental    1.034             0.857            0.749             0.899             0.880              0.857              1.075             0.869              1.787
 hops
 Number of            1.623 ±           1.434 ±          1.047 ±           1.347 ±           1.348 ±            1.190 ±            1.971 ±           1.556 ±            2.502 ±
 inter-country hops   1.166             1.003            0.939             1.014             1.023              0.990              1.372             1.057              3.829

Note. The figure shows the mean values and standard deviations (±) of the target metrics for three approaches: the 𝑛𝑎𝑡𝑖𝑣𝑒 algorithms, the 𝑏𝑎𝑠𝑒𝑙𝑖𝑛𝑒 solution based on static
optimization, and RL-agents trained using the native algorithms (𝑖𝑚𝑝𝑟𝑜𝑣𝑒𝑑). The experiments were repeated five times using the same transaction test set but with different random
seeds, leading to different channel failure probability distributions in each run.


CRediT authorship contribution statement                                                     Appendix B. Multi-criteria comparison details.

    Danila Valko: Writing – review & editing, Writing – original draft,                          See Table 3.
Visualization, Software, Methodology, Data curation, Conceptualiza-
tion. Daniel Kudenko: Writing – review & editing, Writing – original
draft, Validation, Supervision, Conceptualization.
                                                                                             Data availability
Code and data availability
                                                                                                All data and the developed algorithms can be found in the public
   All data and the developed algorithms can be accessed via the public                      repository.
repository: https://github.com/ellariel/ln-dynamic-optimization.

Using generative AI and AI-assisted technologies
                                                                                             References
   During the preparation of this work the authors did not use any
generative AI and AI-assisted technologies in the writing process.                           Alzhrani, F., Saeedi, K., Zhao, L., 2023. Architectural patterns for blockchain sys-
                                                                                                  tems and application design. Appl. Sci. 13 (20), http://dx.doi.org/10.3390/
Declaration of competing interest                                                                 app132011533, [Online]. Available: https://www.mdpi.com/2076-3417/13/20/
                                                                                                  11533.
                                                                                             Asgari, K., Mohammadian, A.A., Tefagh, M., 2022. DyFEn: Agent-based fee setting in
    The authors declare that they have no known competing finan-
                                                                                                  payment channel networks.. http://dx.doi.org/10.48550/ARXIV.2210.08197, arXiv.
cial interests or personal relationships that could have appeared to                         Avarikioti, Z., Heimbach, L., Wang, Y., Wattenhofer, R., 2020. Ride the Lightning: The
influence the work reported in this paper.                                                        game theory of payment channels. In: Financial Cryptography and Data Security:
                                                                                                  24th International Conference, FC 2020 , Kota Kinabalu, Malaysia, February 10–14,
Acknowledgments                                                                                   2020 Revised Selected Papers. Springer-Verlag, Berlin, Heidelberg, pp. 264–283,
                                                                                                  [Online]. Available: https://doi.org/10.1007/978-3-030-51280-4_15.
    We gratefully acknowledge the constructive feedback and insight-                         Avarikioti, Z., Lizurej, T., Michalak, T., Yeo, M., 2023. Lightning creation games.
                                                                                                  In: 2023 IEEE 43rd International Conference on Distributed Computing Systems.
ful suggestions provided by the anonymous reviewers, which greatly
                                                                                                  ICDCS, IEEE Computer Society, Los Alamitos, CA, USA, pp. 1–11. http://dx.doi.
enhanced the quality of this manuscript. We also deeply appreciate                                org/10.1109/ICDCS57875.2023.00037.
the patience and guidance of the editor throughout the review process,                       2024. Azure network round-trip latency statistics.. (Retrieved 15 May 2024) from
ensuring a thorough and thoughtful evaluation of our work.                                        https://learn.microsoft.com/en-us/azure/networking/azure-network-latency.
                                                                                             Barratt, O., Scott, D., 2021. Comparing bitcoin & lightning energy usage to the real
Appendix A. Experiment setup details and hyperparameters                                          world. Retrieved February 22, 2023 from https://blog.coincorner.com/comparing-
                                                                                                  bitcoin-lightning-energy-usage-to-the-real-world-2d64c62b1783.
    The following hyperparameters for the Stable-Baselines3 PPO imple-                       Beres, F., Seres, I.A., Benczur, A.A., 2020. A cryptoeconomic traffic analysis of bitcoin’s
                                                                                                  lightning network. http://dx.doi.org/10.48550/arXiv.1911.09432, arXiv.
mentation (Ppo, 2021) were used: policy type: actor-critic; timesteps
                                                                                             Boyan, J.A., Littman, M.L., 1993. Packet routing in dynamically changing networks:
per epoch: 50,000 for general training; learning rate: 0.000001; dis-
                                                                                                  A reinforcement learning approach. In: NIPS. [Online]. Available: https://api.
count factor (𝛾): 0.99; GAE parameter (𝜆): 0.95; clipping parame-                                 semanticscholar.org/CorpusID:364332.
ter: 0.2; value function coefficient: 0.5; maximum value for gradient                        Brockman, G., Cheung, V., Pettersson, L., Schneider, J., Schulman, J., Tang, J.,
clipping: 0.5; 4 vectorized environments; random seed is fixed.                                   Zaremba, W., 2016. OpenAI gym. http://dx.doi.org/10.48550/arXiv.1606.01540,
    Hardware setup for training. CPU: AMD EPYC 7662 64-Core Proces-                               arXiv.
sor, 256 CPUs; GPU: 2 x A100-PCIE-80 GB; 1 TB RAM, 1007.764 GB                               Cai, Y., Fragkos, G., Tsiropoulou, E.E., Veneris, A., 2020. A truth-inducing sybil resistant
available; Platform system: Linux-5.10.0-15-amd64-x86_64-with-                                    decentralized blockchain oracle. In: 2020 2nd Conference on Blockchain Research
                                                                                                  & Applications for Innovative Networks and Services. BRAINS, pp. 128–135. http:
glibc2.31; Python version: 3.10.6.
                                                                                                  //dx.doi.org/10.1109/BRAINS49436.2020.9223272.
    Hardware setup for experiments. CPU: 11th Gen Intel(R) Core(TM)                          Camilo, G.F., Rebello, G.A.F., de Souza, L.A.C., Campista, M.E.M., Costa, L.H.M.K.,
i5-1135G7 @ 2.40 GHz, 8 CPUs; GPU: 1 x NVIDIA GeForce MX350;                                      2024. ProfitPilot: Enabling rebalancing in payment channel networks through
8 GB RAM, 7.675 GB available; Platform system: Windows-10-                                        profitable cycle creation. IEEE Trans. Netw. Serv. Manag. 21 (3), 3167–3178.
10.0.22621-SP0; Python version: 3.9.13.                                                           http://dx.doi.org/10.1109/TNSM.2024.3361250.


                                                                                        15
D. Valko and D. Kudenko                                                                                                Engineering Applications of Arti cial Intelligence 146 (2025) 110225


Camilo, G.F., Rebello, G.A.F., de Souza, L.A.C., Potop-Butucaru, M.G., de Amorim, M.D.,          Khojasteh, H., Tabatabaei, H., 2021. A survey and taxonomy of blockchain-based
    Campista, M.E.M., Costa, H.M.K., 2022. Topological evolution analysis of payment                 payment channel networks. In: 2021 IEEE High Performance Extreme Computing
    channels in the lightning network. In: 2022 IEEE Latin- American Conference                      Conference. HPEC, pp. 1–8. http://dx.doi.org/10.1109/HPEC49654.2021.9622868.
    on Communications. LATINCOM, pp. 1–6, [Online]. Available:               https://api.        Kumble, S.P., Epema, D., Roos, S., 2021. How lightning’s routing diminishes its
    semanticscholar.org/CorpusID:254533911.                                                          anonymity. In: Proceedings of the 16th International Conference on Availability,
Chaudhry, A.U., Yanikomeroglu, H., 2022. When to crossover from earth to space for                   Reliability and Security. ARES ’21, Association for Computing Machinery, New
    lower latency data communications? IEEE Trans. Aerosp. Electron. Syst. 58 (5),                   York, NY, USA, http://dx.doi.org/10.1145/3465481.3465761.
    3962–3978. http://dx.doi.org/10.1109/TAES.2022.3156087.                                      Kumble, S.P., Roos, S., 2021. Comparative analysis of lightning’s routing clients.
Chen, W., Qiu, X., Cai, Z., Tang, B., Du, L., Zheng, Z., 2024. Graph neural network-                 In: 2021 IEEE International Conference on Decentralized Applications and In-
    enhanced reinforcement learning for payment channel rebalancing. IEEE Trans.                     frastructures. DAPPS, pp. 79–84. http://dx.doi.org/10.1109/DAPPS52256.2021.
    Mob. Comput. 23 (6), 7066–7083. http://dx.doi.org/10.1109/TMC.2023.3328473.                      00014.
Chen, W., Qiu, X., Hong, Z., Zheng, Z., Dai, H.-N., Zhang, J., 2022a. Proactive look-            Lin, J.-H., Primicerio, K., Squartini, T., Decker, C., Tessone, C.J., 2020. Lightning
    ahead control of transaction flows for high-throughput payment channel network.                  network: a second path towards centralisation of the bitcoin economy*. New
    In: Proceedings of the 13th Symposium on Cloud Computing. SoCC ’22, Association                  J. Phys. 22 (8), 083022, [Online]. Available: http://dx.doi.org/10.1088/1367-
    for Computing Machinery, New York, NY, USA, pp. 429–444, [Online]. Available:                    2630/aba062.
    https://doi.org/10.1145/3542929.3563491.                                                     Lind, J., Naor, O., Eyal, I., Kelbert, F., Pietzuch, P., Sirer, E.G., 2019. Teechain: A
                                                                                                     secure payment network with asynchronous blockchain access. arXiv:1707.05454.
Chen, Y., Ran, Y., Zhou, J., Zhang, J., Gong, X., 2022b. MPCN-RP: A routing protocol
                                                                                                     [Online]. Available: https://arxiv.org/abs/1707.05454.
    for blockchain-based multi-charge payment channel networks. IEEE Trans. Netw.
                                                                                                 2024. The lightning network daemon. (Retrieved 15 May 2024) from https://github.
    Serv. Manag. 19 (2), 1229–1242. http://dx.doi.org/10.1109/TNSM.2021.3139019.
                                                                                                     com/lightningnetwork/lnd.
2024. Core Lightning (CLN): A specification compliant lightning network implemen-
                                                                                                 2024. Lightning Network graph: Visualization of nodes and channels. (Retrieved 28
    tation in c.. (Retrieved 15 May 2024) from https://github.com/ElementsProject/
                                                                                                     May 2024) from https://lnrouter.app/graph.
    lightning.                                                                                   Luo, X., Li, P., 2022. Learning-based off-chain transaction scheduling in prioritized
D’Angelo, G., Severini, L., Velaj, Y., 2016. On the maximum betweenness improvement                  payment channel networks. IEEE J. Sel. Areas Commun. 40 (12), 3589–3599.
    problem. Electron. Notes Theor. Comput. Sci. 322, 153–168. http://dx.doi.org/10.                 http://dx.doi.org/10.1109/JSAC.2022.3213333.
    1016/j.entcs.2016.03.011, Proceedings of ICTCS 2015, the 16th Italian Conference             Mahdizadeh, M.S., Bahrak, B., Sayad Haghighi, M., 2023. Decentralizing the lightning
    on Theoretical Computer Science.                                                                 network: a score-based recommendation strategy for the autopilot system. Appl.
Dasaklis, T.K., Malamas, V., 2023. A review of the lightning network’s evolution:                    Netw. Sci. 8, 2364–8228. http://dx.doi.org/10.1007/s41109-023-00602-2.
    Unraveling its present state and the emergence of disruptive digital business                Malavolta, G., Moreno-Sanchez, P., Kate, A., Maffei, M., 2016. SilentWhispers: Enforcing
    models. J. Theor. Appl. Electron. Commer. Res. 18 (3), 1338–1364. http://dx.doi.                 security and privacy in decentralized credit networks.. (Retrieved 15 May 2024)
    org/10.3390/jtaer18030068, [Online]. Available: https://www.mdpi.com/0718-                       from https://eprint.iacr.org/2016/1054.
    1876/18/3/68.                                                                                Malavolta, G., Moreno-Sanchez, P., Kate, A., Maffei, M., Ravi, S., 2017. Concurrency and
Davis, V., Harrison, B., 2022. Learning a scalable algorithm for improving betweenness               privacy with payment-channel networks. In: Proceedings of the 2017 ACM SIGSAC
    in the lightning network. In: 2022 Fourth International Conference on Blockchain                 Conference on Computer and Communications Security. CCS ’17, Association for
    Computing and Applications. BCCA, pp. 119–126. http://dx.doi.org/10.1109/                        Computing Machinery, New York, NY, USA, pp. 455–471, [Online]. Available:
    BCCA55292.2022.9922233.                                                                          https://doi.org/10.1145/3133956.3134096.
Decker, C., 2020. Lightning network research: Topology datasets.. http://dx.doi.org/             Mammeri, Z., 2019. Reinforcement learning based routing in networks: Review and
    10.5281/zenodo.4088530, (Retrieved 15 May 2024) from https://github.com/                         classification of approaches. IEEE Access 7, 55916–55950. http://dx.doi.org/10.
    lnresearch/topology.                                                                             1109/ACCESS.2019.2913776.
Decker, C., Wattenhofer, R., 2015. A fast and scalable payment network with bitcoin              Mayadunna, H., Silva, S.L.D., Wedage, I., Pabasara, S., Rupasinghe, L., Liyanapathi-
    duplex micropayment channels. In: Safety-Critical Systems Symposium. [Online].                   rana, C., Kesavan, K.K., Nawarathna, C.P., Sampath, K.K., 2017. Improving trusted
    Available: https://api.semanticscholar.org/CorpusID:6635264.                                     routing by identifying malicious nodes in a MANET using reinforcement learning.
2024. Eclair (french for lightning) is a scala implementation of the lightning network..             In: 2017 Seventeenth International Conference on Advances in ICT for Emerging
    (Retrieved 15 May 2024) from https://github.com/ACINQ/eclair.                                    Regions. ICTer, pp. 1–8, [Online]. Available: https://api.semanticscholar.org/
Gao, J., Shen, Y., Ito, M., Shiratori, N., 2017. Multi-agent Q-learning aided backpres-              CorpusID:40517242.
    sure routing algorithm for delay reduction. http://dx.doi.org/10.48550/arXiv.1708.           McEnroe, P., Wang, S., Liyanage, M., 2024. Towards latency efficient DRL inference:
    06926, arXiv.                                                                                    Improving UAV obstacle avoidance at the edge through model compression. In:
                                                                                                     2024 IEEE 27th International Conference on Intelligent Transportation Systems
Grunspan, C., Pérez-Marco, R., 2019. Ant routing algorithm for the Lightning Network.
                                                                                                     (ITSC), September 24-27, 2024, Edmonton, Canada.
Gudgeon, L., Moreno-Sanchez, P., Roos, S., McCorry, P., Gervais, A., 2020. SoK:
                                                                                                 Mehar, M.I., Shier, C.L., Giambattista, A., Gong, E., Fletcher, G., Sanayhie, R.,
    Layer-two blockchain protocols. In: Bonneau, J., Heninger, N. (Eds.), Financial
                                                                                                     Kim, H.M., Laskowski, M., 2019. Understanding a revolutionary and flawed grand
    Cryptography and Data Security. Springer International Publishing, Cham, pp.
                                                                                                     experiment in blockchain: The DAO attack. J. Cases Inf. Technol. (JCIT) 21 (1),
    201–226.
                                                                                                     19–32. http://dx.doi.org/10.4018/JCIT.2019010102.
Habib, S.J., Marimuthu, P.N., 2013. Comparing communication protocols within an
                                                                                                 Milicevic, M., Jovanovic, L., Bacanin, N., Zivkovic, M., Jovanovic, D., Antonijevic, M.,
    enterprise network for carbon footprint reduction. Netw. Protoc. Algor. 5, 111–126,
                                                                                                     Savanovic, N., Strumberger, I., 2023. Optimizing long short-term memory by
    [Online]. Available: https://api.semanticscholar.org/CorpusID:37003305.
                                                                                                     improved teacher learning-based optimization for ethereum price forecasting. In:
He, F., 2019. Exploration of distributed image compression and transmission algorithms               Shakya, S., Papakostas, G., Kamel, K.A. (Eds.), Mobile Computing and Sustainable
    for wireless sensor networks. Int. J. Online Biomed. Eng. (IJOE) 15 (01), 143–155.               Informatics. Springer Nature Singapore, Singapore, pp. 125–139.
    http://dx.doi.org/10.3991/ijoe.v15i01.9782.                                                  Miller, A., Bentov, I., Bakshi, S., Kumaresan, R., McCorry, P., 2019. Sprites and
Howell, A., Saber, T., Bendechache, M., 2023. Measuring node decentralisation in                     state channels: Payment networks that go faster than lightning. In: Goldberg, I.,
    blockchain peer to peer networks. Blockchain: Res. Appl. 4 (1), 100109. http:                    Moore, T. (Eds.), Financial Cryptography and Data Security - 23rd International
    //dx.doi.org/10.1016/j.bcra.2022.100109.                                                         Conference, FC 2019, Revised Selected Papers. In: Lecture Notes in Computer
Hussein, Z., Salama, M.A., El-Rahman, S.A., 2023. Evolution of blockchain consensus                  Science (including subseries Lecture Notes in Artificial Intelligence and Lec-
    algorithms: a review on the latest milestones of blockchain consensus algorithms.                ture Notes in Bioinformatics), Springer, Germany, pp. 508–526. http://dx.doi.
    Cybersecurity 6, http://dx.doi.org/10.1186/s42400-023-00163-y.                                   org/10.1007/978-3-030-32101-7_30, Publisher Copyright: © 2019, International
Hüttel, H., Staroveski, V., 2020. Secrecy and authenticity properties of the lightning               Financial Cryptography Association.; 23rd International Conference on Financial
    network protocol. In: Proceedings of the 6th International Conference on Infor-                  Cryptography and Data Security, FC 2019 ; Conference date: 18-02-2019 Through
    mation Systems Security and Privacy - ICISSP. SciTePress, INSTICC, pp. 119–130.                  22-02-2019.
    http://dx.doi.org/10.5220/0008974801190130.                                                  Mizdrakovic, V., Kljajic, M., Zivkovic, M., Bacanin, N., Jovanovic, L., Deveci, M.,
Kadry, H., Gadallah, Y., 2021. A machine learning-based routing technique for off-chain              Pedrycz, W., 2024. Forecasting bitcoin: Decomposition aided long short-term
    transactions in payment channel networks. In: 2021 IEEE International Conference                 memory based time series modeling and its explanation with Shapley val-
    on Smart Internet of Things. SmartIoT, pp. 66–73. http://dx.doi.org/10.1109/                     ues. Knowl.-Based Syst. 299, 112026. http://dx.doi.org/10.1016/j.knosys.2024.
    SmartIoT52359.2021.00020.                                                                        112026, [Online]. Available: https://www.sciencedirect.com/science/article/pii/
Kappos, G., Yousaf, H., Piotrowska, A., Kanjalkar, S., Delgado-Segura, S., Miller, A.,               S0950705124006609.
    Meiklejohn, S., 2021. An empirical analysis of privacy in the lightning network.             Mo, K., Ye, P., Ren, X., Wang, S., Li, W., Li, J., 2024. Security and privacy issues in
    In: Borisov, N., Diaz, C. (Eds.), Financial Cryptography and Data Security. Springer             deep reinforcement learning: Threats and countermeasures. ACM Comput. Surv. 56
    Berlin Heidelberg, Berlin, Heidelberg, pp. 167–186.                                              (6), [Online]. Available: https://doi.org/10.1145/3640312.
Khalil, A.A., Rahman, M.A., Kholidy, H.A., 2023. FAKEY: Fake hashed key attack                   Nayak, S.K., Nayak, S.C., Das, S., 2022. Modeling and forecasting cryptocurrency closing
    on payment channel networks. In: 2023 IEEE Conference on Communications                          prices with rao algorithm-based artificial neural networks: A machine learning ap-
    and Network Security. CNS, pp. 1–9. http://dx.doi.org/10.1109/CNS59707.2023.                     proach. FinTech 1 (1), 47–62. http://dx.doi.org/10.3390/fintech1010004, [Online].
    10288911.                                                                                        Available: https://www.mdpi.com/2674-1032/1/1/4.


                                                                                            16
D. Valko and D. Kudenko                                                                                                  Engineering Applications of Arti cial Intelligence 146 (2025) 110225


Papadis, N., Tassiulas, L., 2023. Deep reinforcement learning-based rebalancing policies          Skrynnik, A., Andreychuk, A., Yakovlev, K., Panov, A., 2022. Pathfinding in stochastic
    for profit maximization of relay nodes in payment channel networks. In: Parda-                    environments: learning vs planning. PeerJ Comput. Sci. 8, e1056. http://dx.doi.
    los, P., Kotsireas, I., Knottenbelt, W.J., Leonardos, S. (Eds.), Mathematical Research            org/10.7717/peerj-cs.1056.
    for Blockchain Economy. Springer Nature Switzerland, Cham, pp. 1–27.                          Song, C., Li, L., Gao, H., 2024. Payment channel fee setting dynamic algorithm based
Pickhardt, R., Richter, S., 2021. Optimally reliable & cheap payment flows on the                     on bilstm-PPO. In: Zhang, J., Sun, N. (Eds.), Third International Conference on
    lightning network. arXiv:2107.05322. [Online]. Available: https://arxiv.org/abs/                  Electronic Information Engineering, Big Data, and Computer Technology (EIBDCT
    2107.05322.                                                                                       2024). Vol. 13181, SPIE, International Society for Optics and Photonics, 1318158,
Pietrzak, K., Salem, I., Schmid, S., Yeo, M., 2021. LightPIR: Privacy-preserving route                [Online]. Available: https://doi.org/10.1117/12.3031220.
    discovery for payment channel networks. In: 2021 IFIP Networking Conference.                  Stadelmann, K., 2023. Op-ed: The state of lightning network in 2023.. (Retrieved 15
    IFIP Networking, pp. 1–9, [Online]. Available: https://api.semanticscholar.org/                   May 2024) from https://cryptoslate.com/the-state-of-lightning-network-in-2023/.
    CorpusID:233204329.                                                                           2024. Real-time Lightning Network statistics. (Retrieved 28 May 2024) from https:
Podiatchev, Y., Orda, A., Rottenstreich, O., 2024. Survivable payment channel networks.               //1ml.com/statistics.
    In: 2024 16th International Conference on COMmunication Systems & NET-                        Sutton, R.S., Barto, A.G., 2018. Reinforcement Learning: An Introduction, second ed.
    workS. COMSNETS, pp. 479–487. http://dx.doi.org/10.1109/COMSNETS59351.                            The MIT Press, Cambridge, [Online]. Available: http://incompleteideas.net/book/
    2024.10426906.                                                                                    the-book-2nd.html.
Poon, J., 2017. Plasma : Scalable autonomous smart contracts. [Online]. Available:                Usman, M., Kliazovich, D., Granelli, F., Bouvry, P., Castoldi, P., 2017. Energy efficiency
    https://api.semanticscholar.org/CorpusID:13266881.                                                of TCP: An analytical model and its application to reduce energy consumption
Poon, J., Dryja, T., 2016. The Bitcoin Lightning Network: Scalable off-chain instant                  of the most diffused transport protocol. Int. J. Commun. Syst. 30 (1), e2934.
    payments. (Retrieved 28 May 2024) from https://lightning.network/lightning-                       http://dx.doi.org/10.1002/dac.2934.
    network-paper.pdf.                                                                            Valko, D., Kudenko, D., 2023. Increasing energy efficiency of bitcoin infrastructure with
2021. The proximal policy optimization algorithm. (Retrieved 15 May 2024) from                        reinforcement learning and one-shot path planning for the lightning network. In:
    https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html.                              Proc. of the Adaptive and Learning Agents Workshop (ALA 2023) At AAMAS 2023,
Prihodko, P., Zhigulin, S.N., Sahno, M., Ostrovskiy, A.B., Osuntokun, O., 2016. Flare                 May 29-30. In: ALA 2023, Cruz, Hayes, Wang, Yates (eds.), London, UK, [Online].
    : An approach to routing in lightning network white paper.. (Retrieved 15 May                     Available: https://alaworkshop2023.github.io/papers/ALA2023_paper_40.pdf.
    2024) from https://bitfury.com/content/downloads/whitepaper_flare_an_approach_                Valko, D., Kudenko, D., 2024. Reducing CO2 emissions in a peer-to-peer distributed
    to_routing_in_lightning_network_7_7_2016.pdf.                                                     payment network: Does geography matter in the lightning network? Comput. Netw.
Qiao, Y., Wu, K., Khabbazian, M., 2024. Non-intrusive balance tomography using                        243, 110297. http://dx.doi.org/10.1016/j.comnet.2024.110297.
    reinforcement learning in the lightning network. ACM Trans. Priv. Secur. 27 (1),              Van Engelshoven, Y., Roos, S., 2021. The merchant: Avoiding payment channel deple-
    [Online]. Available: https://doi.org/10.1145/3639366.                                             tion through incentives. In: 2021 IEEE International Conference on Decentralized
2024. Raiden Network specification. (Retrieved 15 May 2024) from https://raiden-                      Applications and Infrastructures. DAPPS, pp. 59–68. http://dx.doi.org/10.1109/
    network-specification.readthedocs.io/en/latest/index.html.                                        DAPPS52256.2021.00012.
Rebello, G.A.F., Camilo, G.F., Potop-Butucaru, M., Campista, M.E.M., de Amorim, M.D.,             Vishwanath, A., Jalali, F., Hinton, K., Alpcan, T., Ayre, R.W.A., Tucker, R.S., 2015.
    Costa, L.H.M.K., 2022. PCNsim: A flexible and modular simulator for payment                       Energy consumption comparison of interactive cloud-based and local applications.
    channel networks. In: IEEE INFOCOM 2022 - IEEE Conference on Computer                             IEEE J. Sel. Areas Commun. 33 (4), 616–626. http://dx.doi.org/10.1109/JSAC.
    Communications Workshops. INFOCOM WKSHPS, pp. 1–2. http://dx.doi.org/10.                          2015.2393431.
    1109/INFOCOMWKSHPS54753.2022.9798003.                                                         Wang, P., Xu, H., Jin, X., Wang, T., 2019. Flash: Efficient dynamic routing for offchain
2023. River lightning network research report. (Retrieved 15 May 2024) from https:                    networks. In: Proceedings of the 15th International Conference on Emerging
    //river.com/learn/files/river-lightning-report-2023.pdf.                                          Networking Experiments and Technologies. CoNEXT ’19, Association for Computing
Roche, S., 2019. Timelock transactions. (Retrieved 09 September 2024) from https:                     Machinery, New York, NY, USA, pp. 370–381. http://dx.doi.org/10.1145/3359989.
    //bitcoindev.network/guides/bitcoinjs-lib/timelock-transactions/.                                 3365411.
Rohrer, E., Malliaris, J., Tschorsch, F., 2019. Discharged payment channels: Quantifying          Wang, X., Yu, R., Yang, D., Xue, G., Gu, H., Li, Z., Zhou, F., 2024. Fence: Fee-based
    the lightning network’s resilience to topology-based attacks. In: 2019 IEEE Euro-                 online balance-aware routing in payment channel networks. IEEE/ACM Trans.
    pean Symposium on Security and Privacy Workshops. EuroS& pw, pp. 347–356,                         Netw. 32 (2), 1661–1676. http://dx.doi.org/10.1109/TNET.2023.3324136.
    [Online]. Available: https://api.semanticscholar.org/CorpusID:128358911.                      Waugh, F., Holz, R., 2020. An empirical study of availability and reliability properties
Roos, S., Moreno-Sanchez, P., Kate, A., Goldberg, I., 2017. Settling payments fast and                of the bitcoin lightning network. arXiv:2006.14358. [Online]. Available: https:
    private: Efficient decentralized routing for path-based transactions. http://dx.doi.              //arxiv.org/abs/2006.14358.
    org/10.48550/ARXIV.1709.05748, arXiv.                                                         Weintraub, B., Nita-Rotaru, C., Roos, S., 2021. Structural attacks on local routing
Salb, M., Zivkovic, M., Bacanin, N., Chhabra, A., Suresh, M., 2022. Support vector                    in payment channel networks. In: 2021 IEEE European Symposium on Security
    machine performance improvements for cryptocurrency value forecasting by en-                      and Privacy Workshops. EuroS&PW, IEEE, pp. 367–379, [Online]. Available: http:
    hanced Sine cosine algorithm. In: Bansal, J.C., Engelbrecht, A., Shukla, P.K. (Eds.),             //dx.doi.org/10.1109/EuroSPW54576.2021.00046.
    Computer Vision and Robotics. Springer Singapore, Singapore, pp. 527–536.                     Wikipedia, 2024. Lightning network. (Retrieved 15 May 2024) from https://en.
Schulman, J., Wolski, F., Dhariwal, P., Radford, A., Klimov, O., 2017. Proximal policy                wikipedia.org/wiki/Lightning_Network.
    optimization algorithms. http://dx.doi.org/10.48550/ARXIV.1707.06347, arXiv.                  Xiong, H., Chen, M., Wu, C., Zhao, Y., Yi, W., 2022. Research on progress of blockchain
Schwartz, D., Youngs, N., Britto, A., 2014. The ripple protocol consensus algorithm.                  consensus algorithm: A review on recent progress of blockchain consensus al-
    [Online]. Available: https://api.semanticscholar.org/CorpusID:26971000.                           gorithms. Futur. Internet 14 (2), http://dx.doi.org/10.3390/fi14020047, [Online].
Seres, I.A., Gulyás, L., Nagy, D.A., Burcsi, P., 2020. Topological analysis of bitcoin’s              Available: https://www.mdpi.com/1999-5903/14/2/47.
    lightning network. In: Pardalos, P., Kotsireas, I., Guo, Y., Knottenbelt, W. (Eds.),          Xu, X., Weber, I., Staples, M., Zhu, L., Bosch, J., Bass, L., Pautasso, C., Rimba, P.,
    Mathematical Research for Blockchain Economy. Springer International Publishing,                  2017. A taxonomy of blockchain-based systems for architecture design. In: 2017
    Cham, pp. 1–12.                                                                                   IEEE International Conference on Software Architecture. ICSA, pp. 243–252. http:
Shell, B., 2022. How many transactions can the lightning network handle?. (Re-                        //dx.doi.org/10.1109/ICSA.2017.33.
    trieved 28 May 2024) from https://voltage.cloud/blog/bitcoin-education/how-                   Yang, J., He, S., Xu, Y., Chen, L., Ren, J., 2019. A trusted routing scheme using
    many-transactions-can-the-lightning-network-handle.                                               blockchain and reinforcement learning for wireless sensor networks. Sensors 19
Sivaraman, V., Venkatakrishnan, S.B., Ruan, K., Negi, P., Yang, L., Mittal, R., Fanti, G.,            (4), http://dx.doi.org/10.3390/s19040970.
    Alizadeh, M., 2020. High throughput cryptocurrency routing in payment channel                 Zabka, P., Foerster, K.-T., Schmid, S., Decker, C., 2022. Empirical evaluation of nodes
    networks. In: Proceedings of the 17th Usenix Conference on Networked Systems                      and channels of the lightning network. Pervasive Mob. Comput. 83, 101584.
    Design and Implementation. NSDI ’20, USENIX Association, USA, pp. 777–796.                        http://dx.doi.org/10.1016/j.pmcj.2022.101584.




                                                                                             17
