#!/bin/sh
SESSION="vscode`pwd | md5sum | cut -b -3`"
echo $SESSION
tmux attach -d -t $SESSION || tmux new-session -s $SESSION