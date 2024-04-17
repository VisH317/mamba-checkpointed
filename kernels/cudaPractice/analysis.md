# Naive Kernel
- on each block - independent threads
- each thread - K elem multiplications each  with independent fetching

# Warp Kernel
- same K elem multiplications each
- Memory access is parallelized instead

# SRAM Cache Kernel
- each thread does k elem multiplications in a same size for loop
- But the nearby area in the related block is cached making memory access much much faster