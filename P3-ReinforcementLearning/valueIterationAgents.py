# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        # Initialize the value function
        states = self.mdp.getStates()
        iterations = self.iterations

        for state in states:
            self.values[state] = 0.0

        for i in range(iterations):
            newValues = util.Counter()
            for state in states:
                if self.mdp.isTerminal(state):
                    continue
                maxVal = float("-inf")
                for action in self.mdp.getPossibleActions(state):
                    qVal = self.computeQValueFromValues(state, action)
                    if qVal > maxVal:
                        maxVal = qVal
                    newValues[state] = maxVal
            self.values = newValues.copy()


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        q_value = 0

        for next_state, probability in self.mdp.getTransitionStatesAndProbs(state, action):
            reward = self.mdp.getReward(state, action, next_state)
            discount = self.discount
            value = self.values[next_state]
            q_value += probability * (reward + discount * value)

        return q_value
        #util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        possible_actions = self.mdp.getPossibleActions(state)
        best_action = None
        best_q_value = float('-inf')

        for action in possible_actions:
            q_value = self.computeQValueFromValues(state, action)
            if q_value > best_q_value:
                best_action = action
                best_q_value = q_value

        return best_action
        #util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        iteration = self.iterations

        for state in states:
            self.values[state] = 0.0

        for i in range(iteration):
            stateIndex = i % len(states)
            state = states[stateIndex]
            q_values = []

            if not self.mdp.isTerminal(state):
                # maxVal = float("-inf")
                actions = self.mdp.getPossibleActions(state)

                for action in actions:
                    QValue = self.computeQValueFromValues(state, action)
                    q_values.append(QValue)
                    # maxVal = max(maxVal, QValue)
                if len(q_values) > 0:
                    self.values[state] = max(q_values)
                else:
                    self.values[state] = 0

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        mdp = self.mdp
        states = mdp.getStates()
        predecessors = {state: set() for state in states}

        for state in states:
            for action in mdp.getPossibleActions(state):
                for nextState, prob in mdp.getTransitionStatesAndProbs(state, action):
                    if prob > 0:
                        predecessors[nextState].add(state)

        queue = util.PriorityQueue()

        for state in states:
            if not mdp.isTerminal(state):
                diff = abs(
                    self.values[state] - self.computeQValueFromValues(state, self.computeActionFromValues(state)))
                queue.update(state, -diff)

        for i in range(self.iterations):
            if queue.isEmpty():
                break
            state = queue.pop()

            if not mdp.isTerminal(state):
                self.values[state] = self.computeQValueFromValues(state, self.computeActionFromValues(state))

            for p in predecessors[state]:
                if not mdp.isTerminal(p):
                    diff = abs(self.values[p] - self.computeQValueFromValues(p, self.computeActionFromValues(p)))
                    if diff > self.theta:
                        queue.update(p, -diff)
