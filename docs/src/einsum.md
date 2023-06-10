# Einstein Summation

`Tensors` only supports single and pairwise tensors summations.

## Primitive operations

### Index summation

```math
U = \sum_i V_i
```

### Index transposition

```math
U_{ji} = V_{ij}
```

### Diagonal

```math
U_i = V_{ii}
```

### Hadamard or element-wise product

```math
C_i = A \odot B = A_i B_i
```

### Outer or tensor product

```math
C_{ij} = A \otimes B = A_i B_j
```

## Composite operations

### Trace

```math
U = \sum_i V_{ii}
```

### Inner or scalar product

```math
C = A \cdot B = \sum_i A_i B_i
```

### Matrix multiplication

```math
C_{ij} = \sum_k A_{ik} B_{kj}
```
