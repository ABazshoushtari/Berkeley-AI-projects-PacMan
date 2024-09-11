# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        score = 0
        food_dist = [util.manhattanDistance(newPos, food) for food in newFood.asList()]
        if len(food_dist) > 0:
            score += 1.0 / min(food_dist)
        else:
            score += 1
        ghost_dist = [util.manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates]
        for dist, time in zip(ghost_dist, newScaredTimes):
            if time > 0:
                score += dist
            else:
                if dist <= 1:
                    score -= 500
                else:
                    score -= 1 / dist
        score += successorGameState.getScore()
        return score


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        pacmanIndex = 0
        ghostIndices = range(1, gameState.getNumAgents())
        legalActions = gameState.getLegalActions(pacmanIndex)
        bestAction = ''
        bestScore = float('-inf')

        def min_value(gameState, depth, ghostIndex):
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)

            legalActions = gameState.getLegalActions(ghostIndex)
            score = float('inf')

            for action in legalActions:
                successor = gameState.generateSuccessor(ghostIndex, action)
                if ghostIndex == max(ghostIndices):
                    score = min(score, max_value(successor, depth + 1))
                else:
                    score = min(score, min_value(successor, depth, ghostIndex + 1))

            return score

        def max_value(gameState, depth):
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)

            legalActions = gameState.getLegalActions(pacmanIndex)
            score = float("-inf")

            for action in legalActions:
                successor = gameState.generateSuccessor(pacmanIndex, action)
                score = max(score, min_value(successor, depth, ghostIndices[0]))

            return score

        for action in legalActions:
            successor = gameState.generateSuccessor(pacmanIndex, action)
            score = min_value(successor, 0, ghostIndices[0])

            if score > bestScore:
                bestScore = score
                bestAction = action

        return bestAction

        # util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        pacmanIndex = 0
        ghostIndices = range(1, gameState.getNumAgents())
        legalActions = gameState.getLegalActions(pacmanIndex)
        bestAction = ''
        bestScore = float('-inf')
        alpha = float('-inf')
        beta = float('inf')

        def min_value(gameState, depth, ghostIndex, alpha, beta):
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)

            legalActions = gameState.getLegalActions(ghostIndex)
            score = float('inf')

            for action in legalActions:
                successor = gameState.generateSuccessor(ghostIndex, action)
                if ghostIndex == max(ghostIndices):
                    score = min(score, max_value(successor, depth + 1, alpha, beta))
                else:
                    score = min(score, min_value(successor, depth, ghostIndex + 1, alpha, beta))

                if score < alpha:
                    return score

                beta = min(beta, score)

            return score

        def max_value(gameState, depth, alpha, beta):
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)

            legalActions = gameState.getLegalActions(pacmanIndex)
            score = float('-inf')

            for action in legalActions:
                successor = gameState.generateSuccessor(pacmanIndex, action)
                score = max(score, min_value(successor, depth, ghostIndices[0], alpha, beta))

                if score > beta:
                    return score

                alpha = max(alpha, score)

            return score

        for action in legalActions:
            successor = gameState.generateSuccessor(pacmanIndex, action)
            score = min_value(successor, 0, ghostIndices[0], alpha, beta)

            if score > bestScore:
                bestScore = score
                bestAction = action

            if score > beta:
                return bestAction

            alpha = max(alpha, score)

        return bestAction
        # util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        pacmanIndex = 0
        ghostIndices = range(1, gameState.getNumAgents())
        legalActions = gameState.getLegalActions(pacmanIndex)
        bestAction = ''
        bestScore = float('-inf')

        def expected_value(gameState, depth, ghostIndex):
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)

            legalActions = gameState.getLegalActions(ghostIndex)
            score = 0
            prob = 1.0 / len(legalActions)

            for action in legalActions:
                successor = gameState.generateSuccessor(ghostIndex, action)
                if ghostIndex == max(ghostIndices):
                    score += prob * max_value(successor, depth + 1)
                else:
                    score += prob * expected_value(successor, depth, ghostIndex + 1)

            return score

        def max_value(gameState, depth):
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)

            legalActions = gameState.getLegalActions(pacmanIndex)
            score = float("-inf")

            for action in legalActions:
                successor = gameState.generateSuccessor(pacmanIndex, action)
                score = max(score, expected_value(successor, depth, ghostIndices[0]))

            return score

        for action in legalActions:
            successor = gameState.generateSuccessor(pacmanIndex, action)
            score = expected_value(successor, 0, ghostIndices[0])

            if score > bestScore:
                bestScore = score
                bestAction = action

        return bestAction
        # util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
        Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
        evaluation function (question 5).

        DESCRIPTION: This evaluation function gives a higher score to game states where Pacman
        eats more food, gets closer to the closest food, and moves closer to scared ghosts. It
        gives a lower score to game states where Pacman gets closer to non-scared ghosts.

        """

    pacmanPosition = currentGameState.getPacmanPosition()
    foods = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    scaredTimers = [ghostState.scaredTimer for ghostState in ghostStates]
    ghostPositions = currentGameState.getGhostPositions()

    food_weight = 10
    ghost_weight = -1000
    scared_ghost_weight = 200

    score = currentGameState.getScore()

    # Evaluate food
    food_list = foods.asList()
    if food_list:
        closest_food_dist = min([util.manhattanDistance(pacmanPosition, food) for food in food_list])
        if closest_food_dist == 0:
            score += 1000  # Reward for eating food
        else:
            score += food_weight / closest_food_dist

    # Evaluate ghosts
    for ghostState, scaredTimer, ghostPosition in zip(ghostStates, scaredTimers, ghostPositions):
        ghost_dist = util.manhattanDistance(pacmanPosition, ghostPosition)
        if scaredTimer > 0:
            score += scared_ghost_weight / (ghost_dist + 1)
        else:
            if ghost_dist <= 1:
                score += ghost_weight - 500  # Penalize for being too close to a ghost
            else:
                score += ghost_weight / ghost_dist

    return score



# Abbreviation
better = betterEvaluationFunction
