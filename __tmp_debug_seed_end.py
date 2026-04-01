import numpy as np, os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from config import LayoutModeConfig, LogicBoxConfig, TrainingSettingConfig
from envs.FluidEnv import FluidEnv
from envs.Wrapper import FollowerEnv, TaskContext

def parse(layout):
    m=str(layout)
    if m.endswith('_inflow_u_fixed'): return m[:-15]
    if m.endswith('_inflow_u'): return m[:-9]
    if m.endswith('_inflow'): return m[:-7]
    return m

layout=LayoutModeConfig.LAYOUT_MODE
base=parse(layout)
tag=f"{layout}_{TrainingSettingConfig.PATH_TYPE}_{str(LogicBoxConfig.SOURCE_PORT).upper()}_to_{str(LogicBoxConfig.TARGET_PORT).upper()}" if base=='logic_box_layout' else f"{layout}_{TrainingSettingConfig.PATH_TYPE}"
model=os.path.join('models','best',tag,'best_model.zip')
vec=os.path.join('models','best',tag,'vecnormalize_best.pkl')
print('tag',tag,'model',os.path.isfile(model))
ctx=TaskContext(); fe=FluidEnv(start_pos=np.array([-0.025,0.06],dtype=np.float32), layout_mode=layout); env=FollowerEnv(fe,ctx)
ve=DummyVecEnv([lambda: env])
if os.path.isfile(vec):
    ve=VecNormalize.load(vec,ve); ve.training=False; ve.norm_reward=False
m=PPO.load(model, env=ve, device='cpu')
obs=ve.reset(); a,_=m.predict(obs, deterministic=True)
_,r,_,info=ve.step(a); i=info[0]
print('R',float(r[0]),'miss',i.get('logic_miss_ratio'),'col',i.get('logic_collision_ratio'),'wrong',i.get('logic_wrong_side_ratio'),'out',i.get('logic_outlet_error'))
print('min_fwd_vx',LogicBoxConfig.MIN_FORWARD_VX,'x_step',LogicBoxConfig.X_STEP,'max_x_steps',LogicBoxConfig.MAX_X_STEPS)
for idx,e in enumerate(i.get('logic_exits',[])):
    h=e.get('history',[])
    print(idx,'init_vx',e.get('init_vx'),'exited',e.get('exited'),'collision',e.get('collision'),'side',e.get('side'),'len',len(h),'first',h[0] if h else None,'last',h[-1] if h else None)
