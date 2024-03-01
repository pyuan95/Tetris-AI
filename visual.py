import numpy as np
import cv2 as cv
import os
import glob
import sys
import train
import matplotlib.pyplot as plt

SCALE = 30
FPS = 10


def create_mp4(played_boards, out_path):
    """
    Create an MP4 given an list of Tetris Boards
    """
    size = create_frame(played_boards[0].getState()).shape
    print(size)
    out = cv.VideoWriter(
        out_path,
        cv.VideoWriter_fourcc(*"mp4v"),
        FPS,
        (size[1], size[0]),
    )
    for i in range(len(played_boards)):
        out.write(create_frame(played_boards[i].getState()))
    out.release()


def create_frame(board):
    """
    Given a Tetris Board, returns a BGR image in np.array format
    """
    row = len(board)
    col = len(board[0])
    img = np.zeros((row * SCALE, col * SCALE, 3), np.uint8)

    for i in range(22):
        for j in range(10):
            if board[i][j] > 0:
                cv.rectangle(
                    img,
                    (SCALE * j, SCALE * i),
                    (SCALE * (j + 1), SCALE * (i + 1)),
                    (255, 0, 0),
                    -1,
                )
            if board[i][j] < 0:
                cv.rectangle(
                    img,
                    (SCALE * j, SCALE * i),
                    (SCALE * (j + 1), SCALE * (i + 1)),
                    (0, 0, 255),
                    -1,
                )
    return img


# test_boards = [np.random.randint(-1, 2, size=(22, 10)) for _ in range(60)]
# create_mp4(test_boards)
rollout = sys.argv[1]
rollout = train.load_rollout(rollout)
create_mp4(rollout.states, sys.argv[2] if len(sys.argv) >= 3 else "./tetris.mp4")
