from gym.envs.registration import register

register(
    id='BetEnv-v0',
    entry_point='env.env:BetEnv',
)

register(
    id='DeepBetEnv-v0',
    entry_point='env.deepenv:DeepBetEnv',
)

register(
    id='FA_Env-v0',
    entry_point='env.faenv:FA_Env',
)
