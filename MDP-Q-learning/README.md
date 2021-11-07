# Value Iteration (VI) and Policy Iteration (PI) to solve Markov Decision Processes (MDPs) and Q-learning to control a Crawler robot using Python in Jupiter Notebook

## Description

In this project, I implement the two classic methods for solving Markov Decision Processes (MDPs) with finite state and action spaces: Value Iteration (VI) and Policy Iteration (PI), to find the optimal policy in a finite number of iterations. I also implement Q-learning to control a Crawler robot. Implementation details are given below and can also be found in the *MDP-Q-learning.pdf* file. 

## Implementation

*This implementation exercise is adapted from the Berkeley Deep RL Class and the Deep RLvBootcamp held at Berkeley in August 2017.*

In the ﬁrst two problems, you will implement the two classic methods for solving Markov Decision Processes (MDPs) with ﬁnite state and action spaces:

– [**Problem 1**]: Value Iteration (VI).

– [**Problem 2**]: Policy Iteration (PI). 

Both methods find the optimal policy in a ﬁnite number of iterations. The experiments here will use the Frozen Lake environment, a simple gridworld MDP that is taken from the gym package from OpenAI and slightly modiﬁed for this assignment. In this MDP, the agent must navigate from the start state to the goal state on a 4x4 grid, with stochastic transitions.

Both VP and PI require access to an MDP’s dynamics model. This requirement can sometimes be restrictive – for example, if the environment is given as a blackbox physics simulator, then we won’t be able to read oﬀ the whole transition model. In the third problem, you will
implement Q-Learning, which can learn from this type of environments:

– [**Problem 3**]: Sampling-based Tabular Q-Learning.

In the experiments for this problem, you will learn to control a Crawler robot, using the environment already implemented in the gym package.

**Implementation Details**: For this assignment, you will need to use Jupyter Notebook.
Instructions for installing the necessary packages and for activating the deeprlbootcamp environment are included in the **readme.md** ﬁle. Write code only in the ”YOUR CODE HERE” sections in the 3 Notebook ﬁles indicated in bold. Skeleton code and more detailed           instructions are provided in each notebook ﬁle.

<pre>
ml4900/
  hw09/
    code/
      <b>Lab 1 - Problem 1.ipynb
      Lab 1 - Problem 2.ipynb
      Lab 1 - Problem 3.ipynb</b>
      crawler env.py
      frozen lake.py
      misc.py
      environment.yml
      readme.md
