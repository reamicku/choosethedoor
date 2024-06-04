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
                            if self.creatures[i]['currentRoom'] > len(self.rooms)-1:
                                self.creatures[i]['won'] = True
            i += 1

    def countCreaturesInRooms(self) -> list[int]:
        out = []
        for i in range(0, len(self.rooms)+1):
            out.append(0)
        for i, el in enumerate(self.creatures):
            roomId = el['currentRoom']
            out[roomId] += 1
        return out

    def printSimulationState(self):
        creaturesInRooms = self.countCreaturesInRooms()
        for i in range(len(self.rooms)+1, 0, -1):
            if i == len(self.rooms)+1:
                print(f'Exit  \tCreatures: {creaturesInRooms[i-1]}')
            else:
                print(f'Room {i}\tCreatures: {creaturesInRooms[i-1]}')

    def getCreaturesIDsInRoom(self, roomId: int) -> list[int]:
        out = []
        for i, creature in enumerate(self.creatures):
            if creature['currentRoom'] == roomId:
                out.append(i)
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