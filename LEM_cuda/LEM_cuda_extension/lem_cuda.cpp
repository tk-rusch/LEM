#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

std::vector<torch::Tensor> lem_unroll_cuda_forward(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor weights_lin_z,
    torch::Tensor bias,
    torch::Tensor bias_lin_z,
    torch::Tensor initial_z_state,
    torch::Tensor initial_y_state,
    torch::Tensor dt);


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
    torch::Tensor dt);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


std::vector<torch::Tensor> lem_unroll_forward(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor weights_lin_z,
    torch::Tensor bias,
    torch::Tensor bias_lin_z,
    torch::Tensor initial_z_state,
    torch::Tensor initial_y_state,
    torch::Tensor dt) {
  CHECK_INPUT(input);
  CHECK_INPUT(weights);
  CHECK_INPUT(weights_lin_z);
  CHECK_INPUT(bias);
  CHECK_INPUT(bias_lin_z);
  CHECK_INPUT(initial_z_state);
  CHECK_INPUT(initial_y_state);
  CHECK_INPUT(dt);

  return lem_unroll_cuda_forward(input, weights, weights_lin_z, bias, bias_lin_z, initial_z_state, initial_y_state, dt);
}

std::vector<torch::Tensor> lem_unroll_backward(
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
  CHECK_INPUT(grad_y_states);
  CHECK_INPUT(grad_z_states);
  CHECK_INPUT(all_X);
  CHECK_INPUT(all_X2);
  CHECK_INPUT(all_multi_scales);
  CHECK_INPUT(all_lin_new_z_state);
  CHECK_INPUT(weights);
  CHECK_INPUT(weights_lin_z);
  CHECK_INPUT(bias);
  CHECK_INPUT(bias_lin_z);
  CHECK_INPUT(initial_y_state);
  CHECK_INPUT(initial_z_state);
  CHECK_INPUT(dt);

  return lem_unroll_cuda_backward(
      grad_y_states,
      grad_z_states,
      all_X,
      all_X2,
      all_multi_scales,
      all_lin_new_z_state,
      weights,
      weights_lin_z,
      bias,
      bias_lin_z,
      initial_y_state,
      initial_z_state,
      dt);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &lem_unroll_forward, "LEM forward unrolled (CUDA)");
  m.def("backward", &lem_unroll_backward, "LEM backward unrolled (CUDA)");
}
