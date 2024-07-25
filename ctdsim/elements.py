import neurality2
from enum import Enum
import random
import numpy as np
from os import path
from tqdm import tqdm
from datetime import datetime
import pandas as pd


class Creature():
    def __init__(self, inputCount, outputCount) -> None:
        self.inputCount = inputCount
        self.outputCount = outputCount
        self.nn = None
        self.inputValues: np.ndarray = np.zeros(self.inputCount)
        self.outputValues: np.ndarray = np.zeros(self.outputCount)

    def setNeuralNetwork(self, nn: neurality2.NeuralNet) -> bool:
        if nn.n_inputs == self.inputCount and nn.n_outputs == self.outputCount:
            self.nn = nn
            return True
        return False

    def setInputValues(self, array: np.ndarray):
        self.inputValues: np.ndarray = array

    def update(self) -> None:
        if isinstance(self.nn, neurality2.NeuralNet):
            self.nn.set_input(self.inputValues)
            self.nn.forward()
            self.outputValues = self.nn.get_output()

    def getOutputValues(self) -> np.ndarray:
        return self.outputValues


class DoorType(Enum):
    TRAP = -1.0
    FAKE = 0.0
    REAL = 1.0


class Simulation():
    def __init__(self) -> None:
        self.rooms = []
        self.rooms_np = []
        self.rooms_realdoor_index = []
        self.creatures = []
        self.room_size = 0

    def getRoomCount(self): return len(self.rooms)

    def getCreatureCount(self): return len(self.creatures)

    def addRoom(self, trapDoorCount: int, fakeDoorCount: int) -> None:
        """Creates a room. A room always has only 1 real door.

        `trapDoorCount` - specify the amount of trap doors in a room.

        `fakeDoorCount` - specify the amount of fake doors in a room."""

        room = []
        for i in range(0, trapDoorCount):
            room.append(DoorType.TRAP)

        for i in range(0, fakeDoorCount):
            room.append(DoorType.FAKE)

        room.append(DoorType.REAL)
        random.shuffle(room)

        room_np = np.zeros(len(room))
        for i, el in enumerate(room):
            room_np[i] = el.value
        room_realdoor_index = room.index(DoorType.REAL)

        self.rooms.append(room)
        self.rooms_np.append(room_np)
        self.rooms_realdoor_index.append(room_realdoor_index)

    def addCreature(self, creature: Creature, startingRoom=0):
        data = {
            'id': self.getCreatureCount(),
            'creature': creature,
            'currentRoom': startingRoom,
            'won': False,
            'dead': False,
            'confidence': 0.0,
            'confidenceSum': 0.0,
            'decisions': 0,
            'fitness': 0.0
        }
        self.creatures.append(data)
    
    def mutateCreatures(self, mutation_rate: float = 0.0):
        for i, el in enumerate(self.creatures):
            self.creatures[i]['creature'].nn.mutate(mutation_rate=mutation_rate)

    def step(self, chooseDoor=False):
        i = 0

        self.room_size = len(self.rooms[0])
        for el in self.creatures:
            if (not self.creatures[i]['won']) and (not self.creatures[i]['dead']):
                rId = self.creatures[i]['currentRoom']
                self.creatures[i]['creature'].setInputValues(
                    self.rooms_np[rId])
                self.creatures[i]['creature'].update()

                if chooseDoor:
                    output = self.creatures[i]['creature'].getOutputValues()

                    realDoorId = self.rooms_realdoor_index[rId]

                    self.creatures[i]['confidenceSum'] += output[realDoorId][0]
                    self.creatures[i]['decisions'] += 1
                    self.creatures[i]['confidence'] = self.creatures[i]['confidenceSum'] / \
                        self.creatures[i]['decisions']
                    self.creatures[i]['fitness'] = self.creatures[i]['currentRoom'] + \
                        self.creatures[i]['confidence']
                    self.creatures[i]['fitness'] -= 0.01*self.creatures[i]['creature'].nn.n_hidden

                    chosenDoorId = np.argmax(output)
                    if chosenDoorId != realDoorId:
                        if self.rooms[rId][chooseDoor] is DoorType.TRAP:
                            self.creatures[i]['dead'] = True
                    else:
                        if not self.creatures[i]['dead']:
                            # When neural network chooses the real door, it advances to the next room.
                            self.creatures[i]['currentRoom'] += 1
                            if self.creatures[i]['currentRoom'] > self.getRoomCount()-1:
                                self.creatures[i]['won'] = True
                                self.creatures[i]['fitness'] = self.creatures[i]['currentRoom'] + \
                                    self.creatures[i]['confidence']
                                self.creatures[i]['fitness'] -= 0.01*self.creatures[i]['creature'].nn.n_hidden
            i += 1

    def countCreaturesInRooms(self) -> list[int]:
        out = []
        for i in range(0, self.getRoomCount()+1):
            out.append(0)
        for i, el in enumerate(self.creatures):
            roomId = el['currentRoom']
            out[roomId] += 1
        return out

    def printSimulationState(self):
        creaturesInRooms = self.countCreaturesInRooms()
        for i in range(self.getRoomCount()+1, 0, -1):
            n = creaturesInRooms[i-1]
            if i == self.getRoomCount()+1:
                print(f'Exit  \tCreatures: {n}')
            else:
                if n > 0:
                    print(f'Room {i}\tCreatures: {n}')

    def getCreaturesIDsInRoom(self, roomId: int) -> list[int]:
        out = []
        for i, creature in enumerate(self.creatures):
            if creature['currentRoom'] == roomId:
                out.append(i)
        return out

    def getCreaturesIDsInRooms(self) -> list[int]:
        out = []
        for i in range(self.getRoomCount()+1):
            out.append(self.getCreaturesIDsInRoom(i))
        return out

    def getFurthestCreatureIDs(self):
        creaturesInRooms = self.countCreaturesInRooms()
        furthestRoomId = 0

        for i, c in reversed(list(enumerate(creaturesInRooms))):
            if c > 0:
                furthestRoomId = i
                break

        return self.getCreaturesIDsInRoom(furthestRoomId)

    def getCreatureElement(self, creatureId: int):
        return self.creatures[creatureId]

    def getBestCreatureElements(self):
        bestIDs = self.getFurthestCreatureIDs()
        out = []
        for i in bestIDs:
            out.append(self.creatures[i])
        return out

    def getBestNCreatures(self, n: int = 1, skipDead: bool = False):
        fitness_dict = {}
        for i in range(0, len(self.creatures)):
            fitness_dict.update({i: self.creatures[i]['fitness']})
        sorted_fitness_dict = dict(
            sorted(fitness_dict.items(), key=lambda x: x[1], reverse=True))

        out = []
        outN = 0
        for key, value in sorted_fitness_dict.items():
            if skipDead and self.creatures[key]['dead']:
                continue
            out.append(self.creatures[key])
            outN += 1
            if outN >= n:
                break
        return out

    def getCreaturesInRoom(self, min_room: int, max_room: int):
        pass

    def getRoomLayout(self, i: int): return self.rooms[i]

    def getRoomsLayout(self):
        out = []
        for i in range(0, self.getRoomCount()):
            out.append(self.getRoomLayout(i))
        return out

    def getRoomLayoutValues(self, i: int):
        out = []
        for door in self.rooms[i]:
            out.append(door.value)
        return out

    def getRoomsLayoutValues(self):
        out = []
        for i in range(0, self.getRoomCount()):
            out.append(self.getRoomLayoutValues(i))
        return out


class CTDSimulator():
    def __init__(self, settings: dict) -> None:
        self.settings = self.get_default_settings()
        self.settings.update(settings)
        self.statistics ={}

    @staticmethod
    def get_default_settings() -> dict:
        settings = {
            'n_rooms': 10,
            'n_trapdoors': 1,
            'n_fakedoors': 3,
            'n_creatures': 1000,
            'static_room_layout': False,
            'n_generations': 30,
            'n_neuralnet_processing_steps': 4,
            'n_reproduced': 100,
            'mutation_rate': 0.006,
            'n_neuralnet_internal_neurons': 4,
            'n_neuralnet_connection_prob': 0.0,
            'save_results': True,
            'save_network_images': True,
            'save_network_realtime_preview': True,
            'result_directory': 'output',
            'print_generation_info': True,
            'print_debug_info': False,
        }

        return settings
    
    @staticmethod
    def get_time_difference(dt1: datetime, dt2: datetime) -> str:
        diff = abs(dt2 - dt1)

        seconds = diff.seconds
        hours, seconds = divmod(seconds, 3600)
        minutes, seconds = divmod(seconds, 60)

        if hours > 0:
            return f'{hours} hours, {minutes:02} minutes and {seconds:02} seconds'
        elif minutes > 0:
            return f'{minutes} minutes and {seconds:02} seconds'
        else:
            return f'{seconds} seconds'

    def simulate_generation_zero(self) -> None:
        pass

    def simulate_next_generation(self) -> None:
        pass

    def simulate(self) -> None:
        begin_time = datetime.now()
        self.statistics.update({
            'time': {
                'begin': begin_time
            }
        })

        end_time = datetime.now()
        self.statistics.update({
            'time': {
                'end': end_time,
                'elapsed': self.get_time_difference(begin_time, end_time)
            }
        })