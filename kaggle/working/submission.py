def my_agent(observation: Struct, configuration: Struct):
    from random import choice
    return choice([c for c in range(configuration.columns) if observation.board[c] == 0])
