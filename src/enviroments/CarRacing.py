import gym
import client


class CarRacing():

    def __init__(self, remote=None):
        self.name = "Ambiente de pista de carro v0 de OpenAI Gym"
        if remote:
            self.remote = remote
            self.env = client.init(remote)

    def 