# Kurtosis-cuda-pytorch
- CUDA Ops for calculating Kurtosis(or Excess Kurtosis) in pytorch.

  For me, it's ultilized to attention mechanism for saliency detection on high-level feature maps.

- Original pytorch-form code as follow:
```python
  import torch
  
  def excess_kurtosis_cal(x:torch.Tensor, dim:int):
      """
          Calculates excess kurtosis of data 'x' started at dimension 'dim'.
          The dimension started at 'dim' is not keeped.

          Input: 
              x, torch.Tensor(B,C,L)
              dim, int, start_dim for calculation.
      """
      dim_n = len(x.shape)
      if dim_n > dim:
          x = torch.flatten(x,start_dim=dim)
      std, mean = torch.std_mean(x, dim)
      n = torch.Tensor([x.shape[dim]]).to(x.device)
      eps = 1e-6  # for stability

      sample_bias_adjustment = (n - 1) / ((n - 2) * (n - 3))
      kurtosis = sample_bias_adjustment * (
          (n + 1)
          * (
              (torch.sum((x.T - mean.unsqueeze(dim).T).T.pow(4), dim) / n)
              / std.pow(4).clamp(min=eps)
          )
          - 3 * (n - 1)
      )
      return kurtosis
     
```
- CUDA Operator realization:

```C++
To be continued.
```
