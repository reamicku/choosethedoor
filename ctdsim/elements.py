from neurality import NeuralNet
import neurality2
from enum import Enum
import random
import numpy as np


class Creature():
    def __init__(self, inputCount, outputCount) -> None:
        self.inputCount = inputCount
        self.outputCount = outputCount
        self.nn = None
        self.inputValues = []
        self.outputValues = []

    def setNeuralNetwork(self, nn: neurality2.NeuralNet) -> bool:
        if nn.n_inputs == self.inputCount and nn.n_outputs == self.outputCount:
            self.nn = nn
            return True
        return False

    def setInputValues(self, array: np.array):
        self.inputValues = array

    def update(self) -> None:
        if isinstance(self.nn, neurality2.NeuralNet):
            self.nn.set_input(self.inputValues)
            self.nn.forward()
            self.outputValues = self.nn.get_output()

    def getOutputValues(self) -> np.array:
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
        
        room_np = np.zeros((len(room), 1))
        for i, el in enumerate(room): room_np[i] = el.value
        room_realdoor_index = room.index(DoorType.REAL)
        # room = self.rooms[self.creatures[i]['currentRoom']]
        # roomVals = np.zeros((self.room_size, 1))
        # for ir, enumval in enumerate(room):
        #     roomVals[ir] = enumval.value

        self.rooms.append(room)
        self.rooms_np.append(room_np)
        self.rooms_realdoor_index.append(room_realdoor_index)

    def addCreature(self, creature: Creature, startingRoom=0):
        data = {
            'id': self.getCreatureCount(),
            'creature': creature,
            'currentRoom': startingRoom,
            'won': False,
            'dead': False
        }
        self.creatures.append(data)

    def step(self, chooseDoor=False):
        i = 0
        confidence_threshold = 0.38
        
        self.room_size = len(self.rooms[0])
        for el in self.creatures:
            if (not self.creatures[i]['won']) and (not self.creatures[i]['dead']):
                rId = self.creatures[i]['currentRoom']
                self.creatures[i]['creature'].setInputValues(self.rooms_np[rId])
                self.creatures[i]['creature'].update()
                
                if chooseDoor:
                    output = self.creatures[i]['creature'].getOutputValues()

                    realDoorId = self.rooms_realdoor_index[rId]
                    
                    # When neural network chooses the trap door, it kills them.
                    dId = 0
                    for dVal in self.rooms_np[rId]:
                        if dVal == DoorType.TRAP.value:
                            if output[dId] > confidence_threshold:
                                self.creatures[i]['dead'] = True
                                break
                        dId += 1
                    
                    if not self.creatures[i]['dead']:
                        # When neural network chooses the real door, it advances to the next room.
                        if output[realDoorId] > confidence_threshold:
                            self.creatures[i]['currentRoom'] += 1
                            if self.creatures[i]['currentRoom'] > self.getRoomCount()-1:
                                self.creatures[i]['won'] = True
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
                if n>0: print(f'Room {i}\tCreatures: {n}')

    def getCreaturesIDsInRoom(self, roomId: int) -> list[int]:
        out = []
        for i, creature in enumerate(self.creatures):
            if creature['currentRoom'] == roomId:
                out.append(i)
        return out

    def getCreaturesIDsInRooms(self) -> list[int]:
        out = []
        for i in range(self.getRoomCount()+1): out.append(self.getCreaturesIDsInRoom(i))
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
    
    def getBestNCreatures(self, n: int = 1):
        creaturesInRooms = self.getCreaturesIDsInRooms()
        
        out = []
        outN = 0
        for i, room in reversed(list(enumerate(creaturesInRooms))):
            for cId in room:
                out.append(self.creatures[cId])
                outN += 1
                if outN >= n: return out
        return out

    def getCreaturesInRoom(self, min_room: int, max_room: int):
        pass
    
    def getRoomLayout(self, i: int): return self.rooms[i]
    
    def getRoomsLayout(self):
        out = []
        for i in range(0, self.getRoomCount()): out.append(self.getRoomLayout(i))
        return out
    
    def getRoomLayoutValues(self, i: int):
        out = []
        for door in self.rooms[i]:
            out.append(door.value)
        return out

    def getRoomsLayoutValues(self):
        out = []
        for i in range(0, self.getRoomCount()): out.append(self.getRoomLayoutValues(i))
        return out
