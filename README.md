## Simple implementation of Regret Matching Algorithm for Nash Equilibrium computation via repeated self-play


This is simple implementation of regret matching algorithm for Nash Equilibrium computation for two players zero sum games via repeated self-play. This code uses game of Rock-Paper-Scissors as an example

First install numpy
```
pip install -r requirements.txt
```

then run main.py

```python
from regretmatching.rps import RPSPlayer
import numpy as np

a = RPSPlayer()
b = RPSPlayer()
t = 10000
for i in range(0, t):
    a_move = a.move()
    b_move = b.move()
    a.learn_from(a_move)
    b.learn_from(b_move)

_2e = np.round(2 * np.max([a.eps(), b.eps()]), 3)
a_ne = a.current_best_response()
b_ne = b.current_best_response()
print("{0} - nash equilibrium for RPS: {1}, {2}".format(_2e, a_ne, b_ne))
```
