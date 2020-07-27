# Connectx_DQN
Implementation of solving [ConnectX](https://www.kaggle.com/c/connectx) using Deep Q-Networks ([DQN](https://arxiv.org/abs/1312.5602)) as project of Kristian Kersting's Deep Learning course at Technical University of Darmstadt, summer term 2020.

## Requirements
To use the Algorithms will require python3 (>=3.6.5).

* This should install all dependencies and the packages (cloning -> into a directory of choice):
    ```bash
    git clone https://github.com/Janik135/Connectx_DQN.git
    cd Connectx_DQN
    pip install -e .
    ``` 
    
## Usage
### ConnectX
* Train DQN on Kaggle's ConnectX environment:
    ```bash
    python3 path/to/connectx.py --env Qube-v0 --save_path furuta_model.pt --log_name furuta_log
    ```

### Testing
* Test the algorithm using [OpenAI gym](https://github.com/openai/gym) CartPole environment.
    ```bash
    python3 path/to/dqn_cartpole/run.py --name CartPole_dqn --env CartPole-v1 --seed 0 DQN
    ```
   
* Test the algorithm using CartPole environment and giving feedback about learning process.
    ```bash
    python3 path/to/dqn_cartpole/run.py --name CartPole_dqn --env CartPole-v1 --eval --render DQN
    ```
    
Additionally works also for different [OpenAI gym environments](http://gym.openai.com/envs/#classic_control).
