# ConnectX summary

**General**  
Train AI model to play ConnectX game.

**Results**  
My model wins 85% games with random agent and 14% games with Negamax agent (KAGGLE's agent)

**Run this repo**
1. Clone repository.
2. Make virtual env `py venv .env` and activate it `. .env/Scripts/activate.bat`
3. Install requirements `pip install requirements.txt`

**What did I learn & observe?**
1. Basic architecture of `PyTorch` reinforcement learning model.
2. Reward function effects.  

# Summary results
![results_graph](https://i.ibb.co/L5VtYF3/results.png)


# Model and Agent
See whole code in (train notebook)[2_train_model_ipynb]

## `PyTorch` Model
```python
class DQN(nn.Module):

    def __init__(
        self,
        learning_rate: float,
        input_dims: Iterable,
        n_actions: int
    ):
        super(DQN, self).__init__()
        
        # network
        self.input_layer = nn.Linear(*input_dims, 128)
        self.middle_layer = nn.Linear(128, 128)
        self.output_layer = nn.Linear(128, n_actions)
        
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.loss = nn.MSELoss()
        self.device = 'cpu'
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.input_layer(state))
        x = F.relu(self.middle_layer(x))
        actions = self.output_layer(x)

        return actions
```

## Main aspects of training Agent
1. Stores past state, action and reward information.
2. Chooses current action basing on `PyToch` model
3. Runs backpropagation to learn.
