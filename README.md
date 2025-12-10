# The Meaning of Life
## CSC 4631 Final Project
##### Authors: Alex Ewart, Mikhail Filippov

### Project Goal: 
Implement **Novelty Search with Quality (NS-Q)** on **Conway's Game of Life** (CGoL, or alternatevely, Seagull).

### Tasks: 
- Solve Growth Problem: Find states which expand to the largest final cell counts
- Solve Migration Problem: Find states which travel the furthest from the center

### Why: 
CGoL is Turing Complete, meaning the game itself is like a programming language (an inefficient one at that). Furthermore, the state-space of CGoL makes it a perfect candidate for NS-Q, as the large and volatile search-space makes even the smallest change propagate to the end of the simulation; NS-Q's nature to search through novel solutions makes it a prime candidate to search through enormous state-spaces, looking for solutions that are not only valuable, but also different from previous ones.

### Program:
