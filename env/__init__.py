from gym.envs.registration import register

register(
    id='BetEnv-v0',
    entry_point='env.env:BetEnv',
)

register(
    id='BetEnv-v1',
    entry_point='env.env:BetEnv1',
)