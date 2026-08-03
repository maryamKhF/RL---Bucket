# RL/ppo_agent.py


from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor



def build_ppo(
    env,
    cfg,
    seed=42
):


    rl_cfg = cfg["rl"]


    # Monitor wrapper
    env = Monitor(env)



    # Network architecture

    policy_kwargs = {

        "net_arch":
            rl_cfg["hidden_layers"]

    }



    model = PPO(

        policy="MlpPolicy",

        env=env,


        verbose=1,

        seed=seed,


        learning_rate=
            rl_cfg["learning_rate"],


        n_steps=
            rl_cfg["n_steps"],


        batch_size=
            rl_cfg["batch_size"],


        n_epochs=
            rl_cfg["n_epochs"],


        gamma=
            rl_cfg["gamma"],


        gae_lambda=
            rl_cfg["gae_lambda"],


        clip_range=
            rl_cfg["clip_range"],


        ent_coef=
            rl_cfg["ent_coef"],


        policy_kwargs=
            policy_kwargs,


        tensorboard_log=
            rl_cfg.get(
                "tensorboard_log",
                "./logs/"
            ),


        device=
            rl_cfg.get(
                "device",
                "auto"
            )

    )


    return model