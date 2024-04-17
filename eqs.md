# Equations in the SSM

**Changes Made**
- keep a local and global hidden state, the local hidden state generates the A matrix directly
- h_local is used for checkpoints - stored in a memory tensor
- attention is run on memory tensor, output multiplied by dynamic E matrix

**Matrix Defs**
- $A$ - hippo matrix
- $A_{local}$ - dynamic matrix for local hidden state
- $B$ - dynamic input matrix
- $C$ - matmul by hidden state
- $D$ - res connection
- $E$ - elementwise gate for attention

**Equations**
1. $Ad = \exp{(\Delta A)}$
2. Bd - uses euler discretization
3. $h_{t+1} = A * h_t + B * x$
4. $h^{local}_{t+1} = A_{local} * h^{local}_t + B * x$
5. $mem[i] = h_{t+1}^{local}$ if mod step size
6. $Q = W_Qx$, $K=W_k mem$, $V=W_v mem$
7. use the self attention equation here
8. $y_{t+1} = C \cdot h_{t+1} + E \cdot att$
9. $y_{n} = y_{a} + x * D$

**Parallel Scan**
- first: $Bx$, second: $A_2(Bx_1) + Bx_2$, third: $A_3(A_2(Bx_1) + Bx_2) + Bx_3$, fourth: $A_4(A_3(A_2(Bx_1) + Bx_2) + Bx_3) + Bx_4$
- 1-2: $(A_1*A_2, A_2(Bx_1) + Bx_2)$, 3: $(A_3, Bx_3)$ 4: $(A_4, Bx_4)$, 3-4: $(A_3A_4, A_4(Bx_3) + Bx_4)$
- 1-4: $(A_1A_2A_3A_4, A_3A_4(A_2(Bx_1) + Bx_2) + A_4(Bx_3) + Bx_4)$