import random
import copy
from math import sqrt, log
# To train his tree, Kirkby needs to be able to run games as both players. That means
# he needs to implement UCT, which that replaces the play_game loop for learning. Thus
# all these imports
from components.cards import Color, initial_hand
from components.fight import resolve_fight, successful_spy_color
from components.game_status import GameStatus
from components.player import Player

def reconstruct_hand(game,player_color):
    '''
    Calculate a player's hand from the move history stored in the game instance.
    '''
    #TODO optimize
    cards_played=[card[player_color.value-1] for card in game.all_fights]
    return [card for card in initial_hand(game.initial_hands[player_color.name]) if card not in cards_played]

class BRState(object):
    '''
    A wrapper around brave_rats game implementing the interface that mcts expects.
    '''
    def __init__(self, player=None, game=None, spied_card=None):
        self.game = game or GameStatus()
        self.red_player = Player(Color.red, brain_fn=None, hand=None)
        self.blue_player = Player(Color.blue, brain_fn=None, hand=None)
        if player.color == Color.red:
            self.red_player.hand = player.hand
            self.playerToMoveNext = self.red_player
            self.playerJustMoved = self.blue_player
            self.blue_player.hand=reconstruct_hand(game=game,player_color=self.blue_player.color)

        # use these later for imaginary play
        self.red_card = None
        self.blue_card = None
    def Clone(self):
        return copy.deepcopy(self)

    def DoMove(self, move):
        '''
        Play a single card (half of a round)
        '''
        if self.playerToMoveNext == self.red_player:
            self.red_card = move
        else:
            self.blue_card = move
            #round finishes when blue plays, so resolve the fight
            resolve_fight(self.red_card, self.blue_card, self.game)

        #switcheroo
        self.playerToMoveNext, self.playerJustMoved = self.playerJustMoved, self.playerToMoveNext

    def GetMoves(self):
        return self.playerToMoveNext.hand

    def GetResult(self, player):
        '''
        Game result scaled to 1.0
        :param player: Specifies point of view. 1.0 is a win for this player.
        '''
        #Should results be current num points? Does that work with the back propagation?
        if self.game.winner == player:
            return 1.0
        elif self.game.winner == player:
            return 0.0
        else:
            return 0.5 #tie

    def ____repr__(self):
        return self.game.score_summary


class Node:
    """ A node in the game tree. Note wins is always from the viewpoint of playerJustMoved.
        Crashes if state not specified.
    """
    def __init__(self, move = None, parent = None, state = None):
        self.move = move # the move that got us to this node - "None" for the root node
        self.parentNode = parent # "None" for the root node
        self.childNodes = []
        self.wins = 0
        self.visits = 0
        self.untriedMoves = state.GetMoves() # future child nodes
        self.playerJustMoved = state.playerJustMoved # the only part of the state that the Node needs later

    def UCTSelectChild(self):
        """ Use the UCB1 formula to select a child node. Often a constant UCTK is applied so we have
            lambda c: c.wins/c.visits + UCTK * sqrt(2*log(self.visits)/c.visits to vary the amount of
            exploration versus exploitation.
        """
        s = sorted(self.childNodes, key = lambda c: c.wins/c.visits + sqrt(2*log(self.visits)/c.visits))[-1]
        return s

    def AddChild(self, m, s):
        """ Remove m from untriedMoves and add a new child node for this move.
            Return the added child node
        """
        n = Node(move = m, parent = self, state = s)
        #self.untriedMoves.remove(m) #can't do this because m is a different copy of the same move
        #A simpler data structure for cards (even just a list of card names) would improve this
        self.untriedMoves = [move for move in self.untriedMoves if move.value != m.value]
        self.childNodes.append(n)
        return n

    def Update(self, result):
        """ Update this node - one additional visit and result additional wins. result must be from the viewpoint of playerJustmoved.
        """
        self.visits += 1
        self.wins += result

    def __repr__(self):
        return "[M:" + str(self.move) + " W/V:" + str(self.wins) + "/" + str(self.visits) + " U:" + str(self.untriedMoves) + "]"

    def TreeToString(self, indent):
        s = self.IndentString(indent) + str(self)
        for c in self.childNodes:
             s += c.TreeToString(indent+1)
        return s

    def IndentString(self,indent):
        s = "\n"
        for i in range (1,indent+1):
            s += "| "
        return s

    def ChildrenToString(self):
        s = ""
        for c in self.childNodes:
             s += str(c) + "\n"
        return s


def UCT(rootstate, itermax, verbose = False):
    """ Conduct a UCT search for itermax iterations starting from rootstate.
        Return the best move from the rootstate.
        Assumes 2 alternating players (player 1 starts), with game results in the range [0.0, 1.0]."""

    rootnode = Node(state = rootstate)

    for i in range(itermax):
#         print 'starting at root node'
        node = rootnode
        state = rootstate.Clone()

        # Select
        while node.untriedMoves == [] and node.childNodes != []: # node is fully expanded and non-terminal
            node = node.UCTSelectChild()
#             print 'making select move ' + str(node.move)
            state.DoMove(node.move)

        # Expand
        if node.untriedMoves != []: # if we can expand (i.e. state/node is non-terminal)
            m = random.choice(node.untriedMoves)
#             print 'making expand move ' + str(m)
            state.DoMove(m)
            node = node.AddChild(m,state) # add child and descend tree

        # Rollout - this can often be made orders of magnitude quicker using a state.GetRandomMove() function
        while not state.game.is_over: # while state is non-terminal
            m = random.choice(state.GetMoves())
#             print 'making rollout move ' + str(m)
            state.DoMove(m)

        # Backpropagate
        while node != None: # backpropagate from the expanded node and work back to the root node
            node.Update(state.GetResult(node.playerJustMoved)) # state is terminal. Update node with result from POV of node.playerJustMoved
            node = node.parentNode

    # Output some information about the tree - can be omitted
    if (verbose): print rootnode.TreeToString(0)
    else: print rootnode.ChildrenToString()

    return sorted(rootnode.childNodes, key = lambda c: c.visits)[-1].move # return the move that was most visited

def kirkby_brain(player, game, spied_card):
    '''A monte carlo tree search AI.
    Algorithm based on http://mcts.ai/code/python.html .
    Named after Kirkby et al (2013) doi: 10.3109/09553002.2013.791407

    :param player: a Player instance
    :param game: a Game instance, used to look up info about played cards, score, etc.
    :param spied_card: If I successfully played a spy last turn, this is the card that the opponent has revealed and
        will play. Otherwise, None
    :return: a card from my player's hand with which to vanquish my opponent.
    '''
    move = UCT(BRState(player, game, spied_card), itermax = 100, verbose = False)
    return move
