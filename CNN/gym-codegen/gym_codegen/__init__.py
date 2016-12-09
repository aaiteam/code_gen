from gym.envs.registration import register

register(
    id='codegen-v0',
    entry_point='gym_codegen.envs:CodegenEnv',
    timestep_limit=1000,
)
register(
    id='codegen-extrahard-v0',
    entry_point='gym_codegen.envs:CodegenExtraHardEnv',
    timestep_limit=1000,
)
