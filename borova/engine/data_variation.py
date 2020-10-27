import random
import time
from typing import Dict

class Variat(object):

    GeneratedVariables = Dict[str, object]
    TracedValues = []

    def __init__(self):
        self.GeneratedVariables = dict()

    # todo a lot of dry here
    def B(self, id: str, trace: bool = False):
        if self.GeneratedVariables.get('id') != None:
            return self.GeneratedVariables[id]
        generated = bool(random.getrandbits(1))
        self.GeneratedVariables[id] = generated

        if trace == True and id not in self.TracedValues:
            self.TracedValues.append(id)

        return generated


    def R(self, id: str, min, max, trace: bool = False):
        if self.GeneratedVariables.get('id') != None:
            return self.GeneratedVariables[id]

        generated = random.uniform(min, max)
        self.GeneratedVariables[id] = generated

        if trace == True and id not in self.TracedValues:
            self.TracedValues.append(id)

        return generated

    def print_traced_values(self, description: str = ''):
        print(time.strftime("%Y%m%d-%H%M%S") + ' ' + description)
        for val in self.TracedValues:
            print(val + ' -> ' + str(self.GeneratedVariables[val]))