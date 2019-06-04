#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 14:48:19 2019

@author: casper
"""

import itertools, sys
from vispy import app, scene, visuals
from matplotlib.pyplot import subplots, show, draw, pause
from numpy import *
def neighbors(point):
    for comb in itertools.product( *list(range(-1, 2) for _ in point)):
        if any(comb):
            yield tuple((k + i)% j for k, i, j in zip(comb, point, dims))

def advance(board):
    newboard = set()
   # points of interests are neighbors of alive cells
    recalc = board | set(itertools.chain(*map(neighbors, board)))
    for point in recalc:
        count = sum((neighbor in board) for neighbor in neighbors(point))
        if count == 3 or all(count == 2 and point in board):

            newboard.add(point)

    return newboard

class canv:
    def __init__(self, board):
        self.board = board

        # plot ! note the parent parameter

        # build your visuals, that's all
        Scatter3D = scene.visuals.create_visual_node(visuals.MarkersVisual)

        # The real-things : plot using scene
        # build canvas
        canvas = scene.SceneCanvas(keys='interactive', show=True)

        # Add a ViewBox to let the user zoom/rotate
        view = canvas.central_widget.add_view()
        view.camera = 'panzoom'

        view.camera.set_range(x = (0, dims[0]), y = (0, dims[1]))
        p1 = Scatter3D(parent=view.scene)
        p1.set_gl_state('translucent', blend=True, depth_test=True)
        tmp    = array(tuple(board))
        a      = zeros((len(tmp),1))
        print(tmp.shape, a.shape)
        tmp    = hstack((tmp, a))
        p1.set_data(tmp,  symbol='s', size=10,
                    edge_width=0.5, edge_color='blue')
        p1.events.update.connect(self.update)


        self.p1 = p1
        self.view = view
    def update(self, ev):
        try:
            tmp    = array(tuple(self.board ))
            a      = zeros((len(tmp),1))
            tmp    = hstack((tmp, a))
            # print(tmp)
            self.p1.set_data(tmp,   symbol='s', size=5,
                        edge_width=0.5, face_color = 'red', edge_color='red')
            self.board = advance(self.board)

            self.view.update()
        except Exception as e:
            print(e)
dims   = (100, 100)
if __name__ == '__main__':


    glider = array ([(2, 2), (2, 1), (2 ,0), (1, 2), (0, 1)])
    block = array([(0,0), (0,1), (1,0), (1,1)])
    tub   = array([(0, 1), (1, 0), (2,1), (1, 2)])
    block += array([10, 40])
    tub   += array([10,60])
    totup = lambda x: tuple(map(tuple,x))
    board = set(totup(glider)).union ( set(totup(block))).union(totup(tub))
    print(board)
    t = canv(board)
    timer = app.Timer()
    timer.connect(t.update)

    timer.start(0)
    if sys.flags.interactive != 1:
        app.run()