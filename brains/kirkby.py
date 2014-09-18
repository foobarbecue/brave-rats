import random
import copy
from math import sqrt, log
# To train his tree, Kirkby needs to be able to run games as both players. That means
# he needs to implement find_best_move, which that replaces the play_game loop for learning. Thus
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
    return [card for card in initial_hand() if card not in cards_played]

class BRState(object):
    '''
    A wrapper around brave_rats game implementing the interface that mcts expects.
    '''
    def __init__(self, player=None, game=None, spied_card=None):
        self.game = game or GameStatus()
        #Create "imaginary" players for simulation
        self.red_player = Player(Color.red, brain_fn=None)
        self.blue_player = Player(Color.blue, brain_fn=None)
        if player.color == Color.red:
            self.red_player.hand = player.hand
            self.playerToMoveNext = self.red_player
            self.playerJustMoved = self.blue_player
            self.blue_player.hand=reconstruct_hand(game=game,player_color=self.blue_player.color)
        elif player.color == Color.blue:
            self.blue_player.hand = player.hand
            self.playerToMoveNext = self.blue_player
            self.playerJustMoved = self.red_player
            self.red_player.hand=reconstruct_hand(game=game,player_color=self.red_player.color)

        # use these later for imaginary play
        self.red_card = None
        self.blue_card = None
    def clone(self):
        return copy.deepcopy(self)

    def do_move(self, move):
        '''
        Play a single card (half of a round)
        '''
        if self.playerToMoveNext == self.red_player:
            self.red_card = move
        else:
            self.blue_card = move
        if self.red_card and self.blue_card:
            resolve_fight(self.red_card, self.blue_card, self.game)
        self.playerToMoveNext.hand.remove(move)
        # Advance play to next player, unless successful spy has been played.
        # TODO
        self.playerToMoveNext, self.playerJustMoved = self.playerJustMoved, self.playerToMoveNext

    def get_moves(self):
        return self.playerToMoveNext.hand

    def get_result(self, player):
        '''
        Game result scaled to 1.0
        :param player: Specifies point of view. 1.0 is a win for this player.
        '''
        #Should results be current num points? Does that work with the back propagation?
        if not self.game.winner:
            return 0.5
        elif self.game.winner == player.color:
            return 1.0 #win
        else:
            return 0.0 #loss



    def ____repr__(self):
        return self.game.score_summary


class Node(object):
    """ A node in the game tree. Note wins is always from the viewpoint of playerJustMoved.
        Crashes if state not specified.
    """
    def __init__(self, move = None, parent = None, state = None):
        self.move = move # the move that got us to this node - "None" for the root node
        self.parentNode = parent # "None" for the root node
        self.childNodes = []
        self.wins = 0
        self.visits = 0
        self.untriedMoves = state.get_moves() # future child nodes
        self.playerJustMoved = state.playerJustMoved # the only part of the state that the Node needs later

    def select_best_child(self):
        """ Use the UCB1 formula to select a child node. Often a constant UCTK is applied so we have
            lambda c: c.wins/c.visits + UCTK * sqrt(2*log(self.visits)/c.visits to vary the amount of
            exploration versus exploitation.
        """
        s = sorted(self.childNodes, key = lambda c: c.wins/c.visits + sqrt(2*log(self.visits)/c.visits))[-1]
        return s

    def add_child(self, m, s):
        """ Remove m from untriedMoves and add a new child node for this move.
            Return the added child node
        """
        n = Node(move = m, parent = self, state = s)
        #self.untriedMoves.remove(m) #can't do this because m is a different copy of the same move
        #A simpler data structure for cards (even just a list of card names) would improve this
        self.untriedMoves = [move for move in self.untriedMoves if move.value != m.value]
        self.childNodes.append(n)
        return n

    def update(self, result):
        """ update this node - one additional visit and result additional wins. result must be from the viewpoint of playerJustmoved.
        """
        self.visits += 1
        self.wins += result
        #TODO this is for testing
        self.hand1 = self.playerJustMoved.hand

    def __repr__(self):
        return "[M:" + str(self.move) + " W/V:" + str(self.wins) + "/" + str(self.visits) + " U:" + str(self.untriedMoves) + "]"

    def tree_to_string(self, indent):
        s = self.indent_string(indent) + str(self)
        for c in self.childNodes:
            s += c.tree_to_string(indent+1)
        return s

    def indent_string(self,indent):
        s = "\n"
        for i in range (1,indent+1):
            s += "| "
        return s

    def children_to_string(self):
        s = ""
        for c in self.childNodes:
            s += str(c) + "\n"
        return s


def find_best_move(rootstate, itermax, verbose = False):
    """ Conduct a UCT search for itermax iterations starting from rootstate.
        Return the best move from the rootstate."""

    rootnode = Node(state = rootstate)

    for i in range(itermax):
#         print 'starting at root node'
        node = rootnode
        state = rootstate.clone()

        # Select
        while node.untriedMoves == [] and node.childNodes != [] and not state.game.is_over: # node is fully expanded and non-terminal
            node = node.select_best_child()
#             print 'making select move ' + str(node.move)
            state.do_move(node.move)

        # Expand
        if node.untriedMoves != []: # if we can expand (i.e. state/node is non-terminal)
            m = random.choice(node.untriedMoves)
#             print 'making expand move ' + str(m)
            state.do_move(m)
            node = node.add_child(m,state) # add child and descend tree

        # Rollout - this can often be made orders of magnitude quicker using a state.GetRandomMove() function
        while not state.game.is_over: # while state is non-terminal
            m = random.choice(state.get_moves())
#             print 'making rollout move ' + str(m)
            state.do_move(m)
#         print "imaginary game end with {}".format(state.game.all_fights)
#         if state.game.winner:
#             print "{} wins imaginary game".format(state.game.winner.name)
#         else:
#             print "tie"

        # Backpropagate
        while node != None: # backpropagate from the expanded node and work back to the root node
            node.update(state.get_result(node.playerJustMoved)) # state is terminal. update node with result from POV of ownPlayer
            node = node.parentNode

#     if verbose:
#         print rootnode.tree_to_string(0)
#     else:
#         print rootnode.children_to_string()

    return sorted(rootnode.childNodes, key = lambda c: c.visits)[-1].move # return the move that was most visited

def kirkby_brain_fn(player, game, spied_card):
    '''A monte carlo tree search AI.
    Algorithm based on http://mcts.ai/code/python.html .
    Named after Kirkby et al (2013) doi: 10.3109/09553002.2013.791407

    :param player: a Player instance
    :param game: a Game instance, used to look up info about played cards, score, etc.
    :param spied_card: If I successfully played a spy last turn, this is the card that the opponent has revealed and
        will play. Otherwise, None
    :return: a card from my player's hand with which to vanquish my opponent.
    '''
    move = find_best_move(BRState(player, game, spied_card), itermax = 10, verbose = False)
    return move
