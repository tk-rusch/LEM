#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

template <typename scalar_t>
__device__ __forceinline__ scalar_t sigmoid(scalar_t z) {
  return 1.0 / (1.0 + exp(-z));
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t d_sigmoid(scalar_t z) {
  const auto s = sigmoid(z);
  return (1.0 - s) * s;
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t d_tanh(scalar_t z) {
  const auto t = tanh(z);
  return 1 - (t * t);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t elu(scalar_t z, scalar_t alpha = 1.0) {
  return fmax(0.0, z) + fmin(0.0, alpha * (exp(z) - 1.0));
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t d_elu(scalar_t z, scalar_t alpha = 1.0) {
  const auto e = exp(z);
  const auto d_relu = z < 0.0 ? 0.0 : 1.0;
  return d_relu + (((alpha * (e - 1.0)) < 0.0) ? (alpha * e) : 0.0);
}



template <typename scalar_t>
__global__ void lem_z_cell_cuda_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> dt,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> multi_scales,
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> old_z_state,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> new_z_state,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> ms_dt,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> non_lin_y_state) {
  //batch index
  const int n = blockIdx.y;
  // column index
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c < multi_scales.size(2)){
    ms_dt[n][c] = dt[0][0]*sigmoid(multi_scales[n][0][c]);
    non_lin_y_state[n][c] = tanh(multi_scales[n][1][c]);
    new_z_state[n][c] = old_z_state[n][c] + ms_dt[n][c] * (non_lin_y_state[n][c] - old_z_state[n][c]);
  }
}

template <typename scalar_t>
__global__ void lem_y_cell_cuda_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> dt,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> multi_scales,
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> old_y_state,
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> lin_new_z_state,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> new_y_state,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> ms_dt_bar,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> non_lin_z_state) {
  //batch index
  const int n = blockIdx.y;
  // column index
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c < multi_scales.size(2)){
    ms_dt_bar[n][c] = dt[0][0]*sigmoid(multi_scales[n][2][c]);
    non_lin_z_state[n][c] = tanh(lin_new_z_state[n][c]);
    new_y_state[n][c] = old_y_state[n][c] + ms_dt_bar[n][c] * (non_lin_z_state[n][c] - old_y_state[n][c]);
  }
}

std::vector<torch::Tensor> lem_unroll_cuda_forward(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor weights_lin_z,
    torch::Tensor bias,
    torch::Tensor bias_lin_z,
    torch::Tensor initial_z_state,
    torch::Tensor initial_y_state,
    torch::Tensor dt) {

  const auto timesteps = input.size(0);
  const auto input_dim = input.size(2);
  const auto batch_size = initial_z_state.size(0);
  const auto state_size = initial_z_state.size(1);

  auto options = torch::TensorOptions().dtype(input.dtype()).device(input.device().type());
  auto all_y_states = torch::zeros({timesteps, batch_size, state_size}, options);
  auto all_z_states = torch::zeros({timesteps, batch_size, state_size}, options);
  auto all_X = torch::zeros({timesteps, batch_size, state_size + input_dim}, options);
  auto all_X2 = torch::zeros({timesteps, batch_size, state_size + input_dim}, options);
  auto all_multi_scales = torch::zeros({timesteps, batch_size, 3, state_size}, options);
  auto all_lin_new_z_state = torch::zeros({timesteps, batch_size, state_size}, options);

  auto X = torch::zeros({batch_size, state_size + input_dim}, options);
  auto multi_scales_weights = torch::zeros({batch_size, 3 * state_size}, options);
  auto multi_scales = torch::zeros({batch_size, 3, state_size}, options);
  auto new_z_state = torch::zeros_like(initial_z_state);
  auto ms_dt = torch::zeros_like(initial_z_state);
  auto non_lin_y_state = torch::zeros_like(initial_z_state);
  auto new_y_state = torch::zeros_like(initial_z_state);
  auto ms_dt_bar = torch::zeros_like(initial_z_state);
  auto non_lin_z_state = torch::zeros_like(initial_z_state);
  auto old_z_state = initial_z_state;
  auto old_y_state = initial_y_state;

  auto X2 = torch::zeros({batch_size, state_size + input_dim}, options);
  auto lin_new_z_state = torch::zeros({batch_size, state_size}, options);

  const int threads = 1024;
  const dim3 blocks((state_size + threads - 1) / threads, batch_size);

  for (int t=0; t < timesteps; t++) {
    X = torch::cat({old_y_state, input[t]}, /*dim=*/1);
    multi_scales_weights = torch::addmm(bias, X, weights.transpose(0, 1));
    multi_scales = multi_scales_weights.reshape({batch_size, 3, state_size});

    AT_DISPATCH_FLOATING_TYPES(multi_scales.type(), "lem_z_cell_cuda_forward_kernel", ([&] {
    lem_z_cell_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
        dt.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        multi_scales.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
        old_z_state.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        new_z_state.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        ms_dt.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        non_lin_y_state.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>());
    }));

    X2 = torch::cat({new_z_state, input[t]}, /*dim=*/1);
    lin_new_z_state = torch::addmm(bias_lin_z, X2, weights_lin_z.transpose(0, 1));

    AT_DISPATCH_FLOATING_TYPES(multi_scales.type(), "lem_y_cell_cuda_forward_kernel", ([&] {
    lem_y_cell_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
        dt.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        multi_scales.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
        old_y_state.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        lin_new_z_state.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        new_y_state.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        ms_dt_bar.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        non_lin_z_state.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>());
    }));
  all_y_states[t].copy_(new_y_state);
  all_z_states[t].copy_(new_z_state);
  all_X[t].copy_(X);
  all_X2[t].copy_(X2);
  all_multi_scales[t].copy_(multi_scales);
  all_lin_new_z_state[t].copy_(lin_new_z_state);

  old_z_state = new_z_state;
  old_y_state = new_y_state;
  }

  return {all_y_states, all_z_states, all_X, all_X2, all_multi_scales, all_lin_new_z_state};
}



template <typename scalar_t>
__global__ void lem_z_cell_cuda_backward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> dt,
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> d_multi_scales,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> d_z_state,
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> z_state,
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> d_z_new_state,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> multi_scales) {
  //batch index
  const int n = blockIdx.y;
  // column index
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c < d_multi_scales.size(2)){
    d_multi_scales[n][0][c] = d_z_new_state[n][c]*(tanh(multi_scales[n][1][c]) - z_state[n][c])*dt[0][0]*d_sigmoid(multi_scales[n][0][c]);
    d_multi_scales[n][1][c] = d_z_new_state[n][c]*dt[0][0]*sigmoid(multi_scales[n][0][c])*d_tanh(multi_scales[n][1][c]);
    d_z_state[n][c] = d_z_new_state[n][c]*(1.0-dt[0][0]*sigmoid(multi_scales[n][0][c]));
  }
}



template <typename scalar_t>
__global__ void lem_y_cell_cuda_backward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> dt,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> d_lin_new_z_state,
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> d_multi_scales,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> help_d_y_state,
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> grad_y_state,
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> grad_z_state,
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> y_state,
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> lin_new_z_state,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> multi_scales) {
  //batch index
  const int n = blockIdx.y;
  // column index
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c < d_multi_scales.size(2)){
    d_multi_scales[n][2][c] = grad_y_state[n][c]*(tanh(lin_new_z_state[n][c]) - y_state[n][c])*dt[0][0]*d_sigmoid(multi_scales[n][2][c]);
    d_lin_new_z_state[n][c] = grad_y_state[n][c]*dt[0][0]*sigmoid(multi_scales[n][2][c])*d_tanh(lin_new_z_state[n][c]);
    help_d_y_state[n][c] = grad_y_state[n][c]*(1.0-dt[0][0]*sigmoid(multi_scales[n][2][c]));
  }
}

std::vector<torch::Tensor> lem_unroll_cuda_backward(
    torch::Tensor grad_y_states,
    torch::Tensor grad_z_states,
    torch::Tensor all_X,
    torch::Tensor all_X2,
    torch::Tensor all_multi_scales,
    torch::Tensor all_lin_new_z_state,
    torch::Tensor weights,
    torch::Tensor weights_lin_z,
    torch::Tensor bias,
    torch::Tensor bias_lin_z,
    torch::Tensor initial_y_state,
    torch::Tensor initial_z_state,
    torch::Tensor dt) {

  auto options = torch::TensorOptions().dtype(grad_y_states.dtype()).device(grad_y_states.device().type());

  auto d_lin_new_z_state = torch::zeros_like(grad_z_states[-1]);
  auto d_multi_scales = torch::zeros_like(all_multi_scales[-1]);

  auto grad_y_state = torch::zeros_like(grad_y_states[-1]);
  auto grad_z_state = torch::zeros_like(grad_z_states[-1]);
  const auto batch_size = grad_z_state.size(0);
  const auto state_size = grad_z_state.size(1);
  const auto input_dim = all_X2[-1].slice(/*dim=*/1, state_size).size(1);
  const auto timesteps = grad_y_states.size(0);

  auto y_state = torch::zeros_like(all_X[-1].slice(/*dim=*/1, 0, state_size));
  auto old_z_state = torch::zeros_like(all_X2[-1].slice(/*dim=*/1, 0, state_size));
  auto lin_new_z_state = torch::zeros_like(all_lin_new_z_state[-1]);
  auto multi_scales = torch::zeros_like(all_multi_scales[-1]);
  auto X = torch::zeros_like(all_X[-1]);
  auto X2 = torch::zeros_like(all_X2[-1]);

  auto d_weights_lin_z = torch::zeros_like(weights_lin_z);
  auto d_bias_lin_z = torch::zeros_like(bias_lin_z);
  auto d_X2 = torch::zeros_like(X2);
  auto d_z_new_state = torch::zeros_like(y_state);
  auto d_input_2 = torch::zeros({batch_size, input_dim}, options);
  auto d_z_state = torch::zeros_like(d_z_new_state);

  auto d_multi_scales_weights = torch::zeros({batch_size, 3*state_size}, options);
  auto d_weights = torch::zeros_like(weights);
  auto d_bias = torch::zeros_like(bias);
  auto d_X = torch::zeros_like(X);
  auto d_y_state = torch::zeros_like(y_state);
  auto help_d_y_state = torch::zeros_like(y_state);
  auto d_input_1 = torch::zeros({batch_size, input_dim}, options);

  auto d_inputs = torch::zeros({timesteps, batch_size, input_dim}, options);

  const int threads = 1024;
  const dim3 blocks((state_size + threads - 1) / threads, batch_size);

  for (auto t = all_X.size(0) - 1; t>=0; t--) {
    grad_y_state = torch::add(d_y_state,grad_y_states[t]);
    grad_z_state = torch::add(d_z_state,grad_z_states[t]);

    if (t == 0) {
       old_z_state = initial_z_state;
    }
    else {
       old_z_state = all_X2[t-1].slice(/*dim=*/1, 0, state_size);
    }

    y_state = all_X[t].slice(/*dim=*/1, 0, state_size);
    lin_new_z_state = all_lin_new_z_state[t];
    multi_scales = all_multi_scales[t];
    X = all_X[t];
    X2 = all_X2[t];

    AT_DISPATCH_FLOATING_TYPES(X.type(), "lem_y_cell_cuda_backward_kernel", ([&] {
    lem_y_cell_cuda_backward_kernel<scalar_t><<<blocks, threads>>>(
        dt.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        d_lin_new_z_state.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        d_multi_scales.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
        help_d_y_state.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        grad_y_state.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        grad_z_state.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        y_state.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        lin_new_z_state.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        multi_scales.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>());
    }));

    d_weights_lin_z = torch::add(d_weights_lin_z,d_lin_new_z_state.t().mm(X2));
    d_bias_lin_z = torch::add(d_bias_lin_z,d_lin_new_z_state.sum(/*dim=*/0, /*keepdim=*/true));
    d_X2 = d_lin_new_z_state.mm(weights_lin_z);
    d_z_new_state = torch::add(d_X2.slice(/*dim=*/1, 0, state_size),grad_z_state);
    d_input_2 = d_X2.slice(/*dim=*/1, state_size);

    AT_DISPATCH_FLOATING_TYPES(X.type(), "lem_z_cell_cuda_backward_kernel", ([&] {
    lem_z_cell_cuda_backward_kernel<scalar_t><<<blocks, threads>>>(
        dt.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        d_multi_scales.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
        d_z_state.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        old_z_state.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        d_z_new_state.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        multi_scales.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>());
    }));

    d_multi_scales_weights = d_multi_scales.reshape({batch_size, 3*state_size});
    d_weights = torch::add(d_weights,d_multi_scales_weights.t().mm(X));
    d_bias = torch::add(d_bias,d_multi_scales_weights.sum(/*dim=*/0, /*keepdim=*/true));
    d_X = d_multi_scales_weights.mm(weights);
    d_y_state = d_X.slice(/*dim=*/1, 0, state_size);
    d_y_state = torch::add(d_y_state,help_d_y_state);
    d_input_1 = d_X.slice(/*dim=*/1, state_size);
    d_inputs[t] = torch::add(d_input_1,d_input_2);
  }

  return {d_inputs, d_weights, d_weights_lin_z, d_bias, d_bias_lin_z, d_y_state, d_z_state};
}