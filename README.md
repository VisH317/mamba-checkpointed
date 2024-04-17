# Ideas

**The way DNA works**
- requires HUGEEEEEEEEEEEEEEEEEEEEEEEEE context - billions of nucleotides in a genome
- requires LOCAL context - one nucleotide doesnt provide information, but regions can determine introns, exons, promoters, etc.
- Need to find a structure that combiens

**Changes Made**
- Local hidden state along with the long context hidden state - can take relationships over small amounts of nucleotides
  - these checkpoints are constant so they can be cached easily
- `A_local` matrix - used to determine the local hidden state in addition to B
- `h_local` - representation of the current local hidden state (can be calculated in the same way as parallel scan, except with the precomputed A_local mats)
- `memory` - local memory storage at checkpoints
- global attention - combines information across checkpoints to get local information
- `E` matrix - takes the global attention value to add to the output at each value

**What improvements does this make?**
- Mamba has shown promise in long context, but has high perplexity in short context
  - also has problems in RECALL - in this case, recalling promoter, regions (~1000 base pairs) in sequences of millions to billions of nucleotides
  - Its similar to CANINE - while words mean things by themselves, nucleotides and characters only mean something based on their context
- To add short context - we can use based, linear attention is pretty accurate, but quadratic sliding window becomes HELLA EXPENSIVE
  - best to stay sub-quadratic for scaling to longer sequences
- Short context - use an LSTM like approach **(might need to use gating here)**
- long context - same as mamba
- do global checkpoint attention instead
  - attention doesn't work well because individual base pairs don't have meaning, but attention with nearby checkpoints can act as "words"
  - time complexity is quadratic, but based on step size $(n/1000)^2$, can scale to billions of nucleotides with each checkpoint representing little things
- The general idea - Evo uses just stripedhyene with FULL ATTENTION (expensive), and caduceus uses just mamba
  - Combining this to attend to both short context and long context efficiently

**Parallel implementation**
- need to look into parallel scan

**Thing to resaearch** - if you have a smaller vocab size, can you run on fp8 or fp4 more efficiently