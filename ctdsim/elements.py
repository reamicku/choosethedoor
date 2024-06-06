from neurality import NeuralNet
from enum import Enum
import random


class Creature():
    def __init__(self, inputCount, outputCount) -> None:
        self.inputCount = inputCount
        self.outputCount = outputCount
        self.nn = None
        self.inputValues = []
        self.outputValues = []

    def setNeuralNetwork(self, nn: NeuralNet) -> bool:
        if nn.getInputNeuronCount() == self.inputCount and nn.getOutputNeuronCount() == self.outputCount:
            self.nn = nn
            return True
        return False

    def setInputValues(self, array: list[float]):
        if len(array) == self.inputCount:
            self.inputValues = array

    def update(self) -> None:
        if isinstance(self.nn, NeuralNet):
            self.nn.setInputNeuronValues(self.inputValues)
            self.nn.cycle()
            self.outputValues = self.nn.getOutputNeuronValues(useSoftmax=True)

    def getOutputValues(self) -> list[float]:
        return self.outputValues


class DoorType(Enum):
    TRAP = -1.0
    FAKE = 0.0
    REAL = 1.0


class Simulation():
    def __init__(self) -> None:
        self.rooms = []
        self.creatures = []
    
    def getRoomCount(self): return len(self.rooms)
    
    def getCreatureCount(self): return len(self.creatures)

    def addRoom(self, trapDoorCount: int, fakeDoorCount: int, room: list[float] = []) -> None:
        """Creates a room. A room always has only 1 real door.

        `trapDoorCount` - specify the amount of trap doors in a room.

        `fakeDoorCount` - specify the amount of fake doors in a room.

        (optional) `room` - a room layout to add. Ignores previous arguments. """

        if len(room) > 0:
            self.rooms.append(room)
            return

        room = []
        for i in range(0, trapDoorCount):
            room.append(DoorType.TRAP)

        for i in range(0, fakeDoorCount):
            room.append(DoorType.FAKE)

        room.append(DoorType.REAL)
        random.shuffle(room)

        self.rooms.append(room)

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
        for el in self.creatures:
            if (not self.creatures[i]['won']) and (not self.creatures[i]['dead']):
                room = self.rooms[self.creatures[i]['currentRoom']]
                roomVals = []
                for enumval in room:
                    roomVals.append(enumval.value)
                self.creatures[i]['creature'].setInputValues(roomVals)
                self.creatures[i]['creature'].update()
                
                if chooseDoor:
                    output = self.creatures[i]['creature'].getOutputValues()

                    realDoorId = room.index(DoorType.REAL)
                    
                    # When neural network chooses the trap door, it kills them.
                    for dId, d in enumerate(room):
                        if d == DoorType.TRAP:
                            if output[dId] > 0.5:
                                self.creatures[i]['dead'] = True
                                break
                    
                    if not self.creatures[i]['dead']:
                        # When neural network chooses the real door, it advances to the next room.
                        if output[realDoorId] > 0.5:
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
        for i in range(self.getRoomCount()): out.append(self.getCreaturesIDsInRoom(i))
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
