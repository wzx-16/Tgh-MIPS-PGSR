/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include "spatial.h"
#include "simple_knn.h"

torch::Tensor
distCUDA2(const torch::Tensor& points)
{
  const int P = points.size(0);

  auto float_opts = points.options().dtype(torch::kFloat32);
  torch::Tensor means = torch::full({P}, 0.0, float_opts);
  
  SimpleKNN::knn(P, (float3*)points.contiguous().data<float>(), means.contiguous().data<float>());

  return means;
}

torch::Tensor
distCUDA2b(const torch::Tensor& points1, const torch::Tensor& points2)
{
  const int P1 = points1.size(0);
  const int P2 = points2.size(0);

  auto float_opts = points1.options().dtype(torch::kFloat32);
  torch::Tensor means = torch::full({P1, 3}, 0.0, float_opts);


  SimpleKNN::knnb(P1, (float3*)points1.contiguous().data<float>(), P2, (float3*)points2.contiguous().data<float>(), (float3*)means.contiguous().data<float>());
  return means;
}