# On Centralized Critics in Multi-Agent Reinforcement Learning
############################# Folders Description ##############################

Anaconda_Env/ : a virtual env file neurlps21.yml that is going to be used for building dependencies;
multiagent-envs/ : environments' source code;
pg_marl/ : vanilla A2C with a centralized critic (state-based, history-based and state-history-based); 
pymarl_coma_sc/ : COMA with state-based-critic;
pymarl_coma_hc_shc/ : COMA with history-based-critic, and state-history-based-critic;
dectiger/ : for generating probability and value tables for Dec-Tiger
maac/ : MAAC with state-based, history-based and state-history-based critics

############################# Dependencies Installation ##############################
```
cd Anaconda_Env
conda env create -f aaai22.yml 
cd ..

conda activate aaai22

cd multiagent-envs
pip install -e .
cd ..

cd pg_marl
pip install -e .
cd ..

git clone https://github.com/oxwhirl/smac.git
pip install smac/

cd pymarl_coma_sc
bash install_sc2.sh (Note: the user may have to manually modify the path in /pymarl_coma_sc/src/envs/__init__.py in order to find the StarCraftII game)

```
######################### Commends for Reproducing Results  ##################### 

We give the command for one run as an example below. In experiments, We conducted multiple runs, so the user have to specify a run_id for each run in order to save the corresponding result correctly. All results will be stored as a pickle file under /performance/save_dir. 

###### Fig. 3: 2s_vs_1sc ######

SC:
cd pymarl_coma_sc 
python src/main.py --config=coma --env-config=sc2 with env_args.map_name=2s_vs_1sc save_dir=SC_2s_vs_1sc state_critic=True run_id=0

HC:
cd pymarl_coma_hc_shc
python src/main.py --config=coma --env-config=sc2 with env_args.map_name=2s_vs_1sc lr=0.0005 critic_lr=0.0001 target_update_interval=100 save_dir=HC_2s_vs_1sc run_id=0

SHC:
cd pymarl_coma_hc_shc
python src/main.py --config=coma --env-config=sc2 with env_args.map_name=2s_vs_1sc lr=0.0003 critic_lr=0.0001 target_update_interval=1600 state_critic=True  save_dir=SHC_2s_vs_1sc run_id=0

###### Fig. 3: 3m ######

SC:
cd pymarl_coma_sc 
python src/main.py --config=coma --env-config=sc2 with env_args.map_name=3m save_dir=SC_3m state_critic=True run_id=0

HC:
cd pymarl_coma_hc_shc
python src/main.py --config=coma --env-config=sc2 with env_args.map_name=3m lr=0.0005 critic_lr=0.0001 target_update_interval=800 save_dir=HC_3m run_id=0

SHC:
cd pymarl_coma_hc_shc
python src/main.py --config=coma --env-config=sc2 with env_args.map_name=3m lr=0.0005 critic_lr=0.0001 target_update_interval=400 state_critic=True  save_dir=SHC_3m run_id=0

###### Fig. 3: 2s3z ######

SC:
cd pymarl_coma_sc
python src/main.py --config=coma --env-config=sc2 with env_args.map_name=2s3z save_dir=SC_2s3z state_critic=True run_id=0

HC:
cd pymarl_coma_hc_shc
python src/main.py --config=coma --env-config=sc2 with env_args.map_name=2s3z lr=0.0005 critic_lr=0.0001 target_update_interval=400 save_dir=HC_2s3z run_id=0

SHC:
cd pymarl_coma_hc_shc
python src/main.py --config=coma --env-config=sc2 with env_args.map_name=2s3z lr=0.0005 critic_lr=0.0001 target_update_interval=200 state_critic=True  save_dir=SHC_2s3z run_id=0

###### Fig. 3: 1c3s5z ######

SC:
cd pymarl_coma_sc
python src/main.py --config=coma --env-config=sc2 with env_args.map_name=1c3s5z save_dir=SC_1c3s5z state_critic=True run_id=0

HC:
cd pymarl_coma_hc_shc
python src/main.py --config=coma --env-config=sc2 with env_args.map_name=1c3s5z lr=0.0005 critic_lr=0.0001 target_update_interval=200 save_dir=HC_1c3s5z run_id=0

SHC:
cd pymarl_coma_hc_shc
python src/main.py --config=coma --env-config=sc2 with env_args.map_name=1c3s5z lr=0.0005 critic_lr=0.0001 target_update_interval=400 state_critic=True  save_dir=SHC_1c3s5z run_id=0

###### Fig. 3: bane_vs_bane ######

SC:
cd pymarl_coma_sc
python src/main.py --config=coma --env-config=sc2 with env_args.map_name=bane_vs_bane save_dir=SC_bane_vs_bane state_critic=True run_id=0

HC:
cd pymarl_coma_hc_shc
python src/main.py --config=coma --env-config=sc2 with env_args.map_name=bane_vs_bane lr=0.0005 critic_lr=0.0001 target_update_interval=1200 save_dir=HC_bane_vs_bane run_id=0

SHC:
cd pymarl_coma_hc_shc
python src/main.py --config=coma --env-config=sc2 with env_args.map_name=bane_vs_bane lr=0.0003 critic_lr=0.0001 target_update_interval=800 state_critic=True  save_dir=SHC_bane_vs_bane run_id=0

###### Speaker and Listener ######

cd pg_marl

## obs_radius: 0.8

SC:
maac_cenV_agrnn.py with discrete_mul=2 c_target_update_freq=8 n_step_bootstrap=3 total_epies=100_000 n_envs=2 env_name=pomdp_simple_speaker_listener max_epi_steps=25 a_lr=0.0005 c_lr=0.003 train_freq=2 eps_decay_epis=20_000 eps_start=1.0 eps_end=0.05 obs_r=0.8 c_target eval_policy save_dir=SC_SL08 run_idx=0 

HC:
maac_cenV_rnn.py with discrete_mul=2 c_target_update_freq=8 n_step_bootstrap=3 total_epies=100_000 n_envs=2 env_name=pomdp_simple_speaker_listener max_epi_steps=25 a_lr=0.0005 c_lr=0.003 train_freq=2 eps_decay_epis=20_000 eps_start=1.0 eps_end=0.05 obs_r=0.8 c_target eval_policy save_dir=SC_SL08 run_idx=0 

SHC:
maac_cenV_shc.py with discrete_mul=2 c_target_update_freq=64 n_step_bootstrap=3 total_epies=100_000 n_envs=4 env_name=pomdp_simple_speaker_listener max_epi_steps=25 a_lr=0.0005 c_lr=0.0001 train_freq=4 eps_decay_epis=20_000 eps_start=1.0 eps_end=0.05 obs_r=0.8 c_target eval_policy save_dir=SC_SL08 run_idx=0 

## obs_radius: 1.2

SC:
maac_cenV_agrnn.py with discrete_mul=2 c_target_update_freq=16 n_step_bootstrap=3 total_epies=100_000 n_envs=2 env_name=pomdp_simple_speaker_listener max_epi_steps=25 a_lr=0.0005 c_lr=0.001 train_freq=2 eps_decay_epis=20_000 eps_start=1.0 eps_end=0.05 obs_r=1.2 c_target eval_policy save_dir=SC_SL12 run_idx=0 

HC:
maac_cenV_rnn.py with discrete_mul=2 c_target_update_freq=8 n_step_bootstrap=5 total_epies=100_000 n_envs=2 env_name=pomdp_simple_speaker_listener max_epi_steps=25 a_lr=0.0005 c_lr=0.003 train_freq=2 eps_decay_epis=20_000 eps_start=1.0 eps_end=0.05 obs_r=1.2 c_target eval_policy save_dir=SC_SL12 run_idx=0 

SHC:
maac_cenV_shc.py with discrete_mul=2 c_target_update_freq=64 n_step_bootstrap=3 total_epies=100_000 n_envs=2 env_name=pomdp_simple_speaker_listener max_epi_steps=25 a_lr=0.0005 c_lr=0.0001 train_freq=2 eps_decay_epis=20_000 eps_start=1.0 eps_end=0.05 obs_r=0.8 c_target eval_policy save_dir=SC_SL12 run_idx=0 

## obs_radius: 1.6

SC:
maac_cenV_agrnn.py with discrete_mul=2 c_target_update_freq=8 n_step_bootstrap=3 total_epies=100_000 n_envs=2 env_name=pomdp_simple_speaker_listener max_epi_steps=25 a_lr=0.0005 c_lr=0.003 train_freq=2 eps_decay_epis=20_000 eps_start=1.0 eps_end=0.05 obs_r=1.6 c_target eval_policy save_dir=SC_SL16 run_idx=0 

HC:
maac_cenV_rnn.py with discrete_mul=2 c_target_update_freq=8 n_step_bootstrap=5 total_epies=100_000 n_envs=2 env_name=pomdp_simple_speaker_listener max_epi_steps=25 a_lr=0.0005 c_lr=0.003 train_freq=2 eps_decay_epis=20_000 eps_start=1.0 eps_end=0.05 obs_r=1.6 c_target eval_policy save_dir=SC_SL16 run_idx=0 

SHC:
maac_cenV_shc.py with discrete_mul=2 c_target_update_freq=64 n_step_bootstrap=3 total_epies=100_000 n_envs=2 env_name=pomdp_simple_speaker_listener max_epi_steps=25 a_lr=0.0005 c_lr=0.0001 train_freq=2 eps_decay_epis=20_000 eps_start=1.0 eps_end=0.05 obs_r=1.6 c_target eval_policy save_dir=SC_SL16 run_idx=0 

## obs_radius: 2.0

SC:
maac_cenV_agrnn.py with discrete_mul=2 c_target_update_freq=32 n_step_bootstrap=3 total_epies=100_000 n_envs=2 env_name=pomdp_simple_speaker_listener max_epi_steps=25 a_lr=0.0005 c_lr=0.003 train_freq=2 eps_decay_epis=20_000 eps_start=1.0 eps_end=0.05 obs_r=2.0 c_target eval_policy save_dir=SC_SL20 run_idx=0 

HC:
maac_cenV_rnn.py with discrete_mul=2 c_target_update_freq=16 n_step_bootstrap=3 total_epies=100_000 n_envs=2 env_name=pomdp_simple_speaker_listener max_epi_steps=25 a_lr=0.0005 c_lr=0.003 train_freq=2 eps_decay_epis=20_000 eps_start=1.0 eps_end=0.05 obs_r=2.0 c_target eval_policy save_dir=SC_SL20 run_idx=0 

SHC:
maac_cenV_shc.py with discrete_mul=2 c_target_update_freq=64 n_step_bootstrap=3 total_epies=100_000 n_envs=2 env_name=pomdp_simple_speaker_listener max_epi_steps=25 a_lr=0.0005 c_lr=0.0001 train_freq=2 eps_decay_epis=20_000 eps_start=1.0 eps_end=0.05 obs_r=2.0 c_target eval_policy save_dir=SC_SL20 run_idx=0 

## obs_radius: 2.4

SC:
maac_cenV_agrnn.py with discrete_mul=2 c_target_update_freq=8 n_step_bootstrap=3 total_epies=100_000 n_envs=2 env_name=pomdp_simple_speaker_listener max_epi_steps=25 a_lr=0.0005 c_lr=0.003 train_freq=2 eps_decay_epis=20_000 eps_start=1.0 eps_end=0.05 obs_r=2.4 c_target eval_policy save_dir=SC_SL24 run_idx=0 

HC:
maac_cenV_rnn.py with discrete_mul=2 c_target_update_freq=8 n_step_bootstrap=5 total_epies=100_000 n_envs=2 env_name=pomdp_simple_speaker_listener max_epi_steps=25 a_lr=0.0005 c_lr=0.003 train_freq=2 eps_decay_epis=20_000 eps_start=1.0 eps_end=0.05 obs_r=2.4 c_target eval_policy save_dir=SC_SL24 run_idx=0 

SHC:
maac_cenV_shc.py with discrete_mul=2 c_target_update_freq=64 n_step_bootstrap=5 total_epies=100_000 n_envs=4 env_name=pomdp_simple_speaker_listener max_epi_steps=25 a_lr=0.0005 c_lr=0.0003 train_freq=4 eps_decay_epis=20_000 eps_start=1.0 eps_end=0.05 obs_r=2.4 c_target eval_policy save_dir=SC_SL24 run_idx=0 


###### Cooperative Navigation ######

cd pg_marl

## obs_radius: 0.8

SC:
maac_cenV_agrnn.py with discrete_mul=2 c_target_update_freq=8 n_step_bootstrap=5 total_epies=100_000 n_envs=2 env_name=pomdp_simple_spread max_epi_steps=25 a_lr=0.0005 c_lr=0.003 train_freq=2 eps_decay_epis=50_000 eps_start=1.0 eps_end=0.05 obs_r=0.8 c_target eval_policy save_dir=SC_CN08 run_idx=0 

HC:
maac_cenV_rnn.py with discrete_mul=2 c_target_update_freq=16 n_step_bootstrap=5 total_epies=100_000 n_envs=2 env_name=pomdp_simple_spread max_epi_steps=25 a_lr=0.0005 c_lr=0.003 train_freq=2 eps_decay_epis=50_000 eps_start=1.0 eps_end=0.05 obs_r=0.8 c_target eval_policy save_dir=SC_CN08 run_idx=0 

SHC:
maac_cenV_shc.py with discrete_mul=2 c_target_update_freq=32 n_step_bootstrap=5 total_epies=100_000 n_envs=2 env_name=pomdp_simple_spread max_epi_steps=25 a_lr=0.0005 c_lr=0.0005 train_freq=2 eps_decay_epis=50_000 eps_start=1.0 eps_end=0.05 obs_r=0.8 c_target eval_policy save_dir=SC_CN08 run_idx=0 

## obs_radius: 1.0

SC:
maac_cenV_agrnn.py with discrete_mul=2 c_target_update_freq=16 n_step_bootstrap=3 total_epies=100_000 n_envs=2 env_name=pomdp_simple_spread max_epi_steps=25 a_lr=0.001 c_lr=0.001 train_freq=2 eps_decay_epis=50_000 eps_start=1.0 eps_end=0.05 obs_r=1.0 c_target eval_policy save_dir=SC_CN10 run_idx=0 

HC:
maac_cenV_rnn.py with discrete_mul=2 c_target_update_freq=16 n_step_bootstrap=5 total_epies=100_000 n_envs=2 env_name=pomdp_simple_spread max_epi_steps=25 a_lr=0.0005 c_lr=0.003 train_freq=2 eps_decay_epis=50_000 eps_start=1.0 eps_end=0.05 obs_r=1.0 c_target eval_policy save_dir=SC_CN10 run_idx=0 

SHC:
maac_cenV_shc.py with discrete_mul=2 c_target_update_freq=32 n_step_bootstrap=5 total_epies=100_000 n_envs=2 env_name=pomdp_simple_spread max_epi_steps=25 a_lr=0.0005 c_lr=0.0005 train_freq=2 eps_decay_epis=50_000 eps_start=1.0 eps_end=0.05 obs_r=1.0 c_target eval_policy save_dir=SC_CN10 run_idx=0 

## obs_radius: 1.2

SC:
maac_cenV_agrnn.py with discrete_mul=2 c_target_update_freq=8 n_step_bootstrap=3 total_epies=100_000 n_envs=2 env_name=pomdp_simple_spread max_epi_steps=25 a_lr=0.001 c_lr=0.001 train_freq=2 eps_decay_epis=50_000 eps_start=1.0 eps_end=0.05 obs_r=1.2 c_target eval_policy save_dir=SC_CN12 run_idx=0 

HC:
maac_cenV_rnn.py with discrete_mul=2 c_target_update_freq=16 n_step_bootstrap=5 total_epies=100_000 n_envs=2 env_name=pomdp_simple_spread max_epi_steps=25 a_lr=0.0005 c_lr=0.003 train_freq=2 eps_decay_epis=50_000 eps_start=1.0 eps_end=0.05 obs_r=1.2 c_target eval_policy save_dir=SC_CN12 run_id=0 

SHC:
maac_cenV_shc.py with discrete_mul=2 c_target_update_freq=16 n_step_bootstrap=5 total_epies=100_000 n_envs=2 env_name=pomdp_simple_spread max_epi_steps=25 a_lr=0.0005 c_lr=0.0005 train_freq=2 eps_decay_epis=50_000 eps_start=1.0 eps_end=0.05 obs_r=1.2 c_target eval_policy save_dir=SC_CN12 run_id=0 

## obs_radius: 1.4

SC:
maac_cenV_agrnn.py with discrete_mul=2 c_target_update_freq=8 n_step_bootstrap=3 total_epies=100_000 n_envs=2 env_name=pomdp_simple_spread max_epi_steps=25 a_lr=0.001 c_lr=0.001 train_freq=2 eps_decay_epis=50_000 eps_start=1.0 eps_end=0.05 obs_r=1.4 c_target eval_policy save_dir=SC_CN14 run_idx=0 

HC:
maac_cenV_rnn.py with discrete_mul=2 c_target_update_freq=8 n_step_bootstrap=5 total_epies=100_000 n_envs=2 env_name=pomdp_simple_spread max_epi_steps=25 a_lr=0.0005 c_lr=0.003 train_freq=2 eps_decay_epis=50_000 eps_start=1.0 eps_end=0.05 obs_r=1.4 c_target eval_policy save_dir=SC_CN14 run_idx=0 

SHC:
maac_cenV_shc.py with discrete_mul=2 c_target_update_freq=32 n_step_bootstrap=5 total_epies=100_000 n_envs=2 env_name=pomdp_simple_spread max_epi_steps=25 a_lr=0.0005 c_lr=0.0003 train_freq=2 eps_decay_epis=50_000 eps_start=1.0 eps_end=0.05 obs_r=1.4 c_target eval_policy save_dir=SC_CN14 run_idx=0 
