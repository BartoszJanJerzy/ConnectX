def my_agent(observation, configuration):
    import os
    import numpy as np
    import torch as T
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim


    class AgentModel(nn.Module):
    
        def __init__(self):
            super(AgentModel, self).__init__()
            self.input_layer = nn.Linear(44, 128)
            self.middle_layer = nn.Linear(128, 128) 
            self.output_layer = nn.Linear(128, 7) 
            self.optimizer = optim.Adam(self.parameters(), lr=0.003)
            self.loss = nn.MSELoss()
            self.device = 'cpu'
            self.to(self.device)

        def load_action(
            self,
            observation_input: list,
            board: list,
            rows: int,
            cols: int
        ) -> int:
            state = T.tensor(observation_input, dtype=T.float32).to('cpu')
            actions = self.forward(state)

            board = np.array(board).reshape(rows, cols).T
            base_actions_list: list = actions.tolist()
            final_actions_list: list = actions.tolist()      
            actions_dict = {k: v for k, v in zip(base_actions_list, range(len(base_actions_list)))}
            
            for i in range(cols):
                if board[i][0]:
                    final_actions_list.remove(base_actions_list[i])
            
            if len(final_actions_list):
                action = actions_dict[max(final_actions_list)]
            else:
                action = 0
            
            return action

        def forward(self, state) -> T.tensor:
            x = F.relu(self.input_layer(state))
            x = F.relu(self.middle_layer(x))
            actions = self.output_layer(x)

            return actions

    weights_path = os.path.join(os.getcwd(), 'kaggle', 'working', 'weights')
    if not os.path.isfile(weights_path):
        weights_path = os.path.join(os.getcwd(), 'weights')
    
    model = AgentModel()
    model.load_state_dict(T.load(weights_path))

    if configuration is None:
        rows=6
        cols=7
    else:
        rows = configuration.rows
        cols = configuration.columns
        
    observation_input= observation.board + [observation.step, observation.mark]
    action = model.load_action(observation_input, observation.board, rows, cols)

    return action
