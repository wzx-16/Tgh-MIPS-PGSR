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

#define BOX_SIZE 1024

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "simple_knn.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <vector>
#include <cuda_runtime_api.h>
#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#define __CUDACC__
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <float.h>
namespace cg = cooperative_groups;

struct CustomMin
{
	__device__ __forceinline__
		float3 operator()(const float3& a, const float3& b) const {
		return { min(a.x, b.x), min(a.y, b.y), min(a.z, b.z) };
	}
};

struct CustomMax
{
	__device__ __forceinline__
		float3 operator()(const float3& a, const float3& b) const {
		return { max(a.x, b.x), max(a.y, b.y), max(a.z, b.z) };
	}
};

__host__ __device__ uint32_t prepMorton(uint32_t x)
{
	x = (x | (x << 16)) & 0x030000FF;
	x = (x | (x << 8)) & 0x0300F00F;
	x = (x | (x << 4)) & 0x030C30C3;
	x = (x | (x << 2)) & 0x09249249;
	return x;
}

__host__ __device__ uint32_t coord2Morton(float3 coord, float3 minn, float3 maxx)
{
	uint32_t x = prepMorton(((coord.x - minn.x) / (maxx.x - minn.x)) * ((1 << 10) - 1));
	uint32_t y = prepMorton(((coord.y - minn.y) / (maxx.y - minn.y)) * ((1 << 10) - 1));
	uint32_t z = prepMorton(((coord.z - minn.z) / (maxx.z - minn.z)) * ((1 << 10) - 1));

	return x | (y << 1) | (z << 2);
}

__global__ void coord2Morton(int P, const float3* points, float3 minn, float3 maxx, uint32_t* codes)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	codes[idx] = coord2Morton(points[idx], minn, maxx);
}

struct MinMax
{
	float3 minn;
	float3 maxx;
};

__global__ void boxMinMax(uint32_t P, float3* points, uint32_t* indices, MinMax* boxes)
{
	auto idx = cg::this_grid().thread_rank();

	MinMax me;
	if (idx < P)
	{
		me.minn = points[indices[idx]];
		me.maxx = points[indices[idx]];
	}
	else
	{
		me.minn = { FLT_MAX, FLT_MAX, FLT_MAX };
		me.maxx = { -FLT_MAX,-FLT_MAX,-FLT_MAX };
	}

	__shared__ MinMax redResult[BOX_SIZE];

	for (int off = BOX_SIZE / 2; off >= 1; off /= 2)
	{
		if (threadIdx.x < 2 * off)
			redResult[threadIdx.x] = me;
		__syncthreads();

		if (threadIdx.x < off)
		{
			MinMax other = redResult[threadIdx.x + off];
			me.minn.x = min(me.minn.x, other.minn.x);
			me.minn.y = min(me.minn.y, other.minn.y);
			me.minn.z = min(me.minn.z, other.minn.z);
			me.maxx.x = max(me.maxx.x, other.maxx.x);
			me.maxx.y = max(me.maxx.y, other.maxx.y);
			me.maxx.z = max(me.maxx.z, other.maxx.z);
		}
		__syncthreads();
	}

	if (threadIdx.x == 0)
		boxes[blockIdx.x] = me;
}

__device__ __host__ float distBoxPoint(const MinMax& box, const float3& p)
{
	float3 diff = { 0, 0, 0 };
	if (p.x < box.minn.x || p.x > box.maxx.x)
		diff.x = min(abs(p.x - box.minn.x), abs(p.x - box.maxx.x));
	if (p.y < box.minn.y || p.y > box.maxx.y)
		diff.y = min(abs(p.y - box.minn.y), abs(p.y - box.maxx.y));
	if (p.z < box.minn.z || p.z > box.maxx.z)
		diff.z = min(abs(p.z - box.minn.z), abs(p.z - box.maxx.z));
	return diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;
}

template<int K>
__device__ void updateKBest(const float3& ref, const float3& point, float* knn)
{
	float3 d = { point.x - ref.x, point.y - ref.y, point.z - ref.z };
	float dist = d.x * d.x + d.y * d.y + d.z * d.z;
	for (int j = 0; j < K; j++)
	{
		if (knn[j] > dist)
		{
			float t = knn[j];
			knn[j] = dist;
			dist = t;
		}
	}
}

template<int K>
__device__ void updateKBestv(const float3& ref, const float3& point, float* knn, float3* knnxyz)
{
	float3 d = { point.x - ref.x, point.y - ref.y, point.z - ref.z };
	float dist = d.x * d.x + d.y * d.y + d.z * d.z;
	float3 p = point;
	for (int j = 0; j < K; j++)
	{
		if (knn[j] > dist)
		{
			float t = knn[j];
			knn[j] = dist;
			dist = t;
			float3 p_tmp = knnxyz[j];
			knnxyz[j] = p;
			p = p_tmp;
		}
	}
}

__global__ void boxMeanDist(uint32_t P, float3* points, uint32_t* indices, MinMax* boxes, float* dists)
{
	int idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	float3 point = points[indices[idx]];
	float best[3] = { FLT_MAX, FLT_MAX, FLT_MAX };

	for (int i = max(0, idx - 3); i <= min(P - 1, idx + 3); i++)
	{
		if (i == idx)
			continue;
		updateKBest<3>(point, points[indices[i]], best);
	}

	float reject = best[2];
	best[0] = FLT_MAX;
	best[1] = FLT_MAX;
	best[2] = FLT_MAX;

	for (int b = 0; b < (P + BOX_SIZE - 1) / BOX_SIZE; b++)
	{
		MinMax box = boxes[b];
		float dist = distBoxPoint(box, point);
		if (dist > reject || dist > best[2])
			continue;

		for (int i = b * BOX_SIZE; i < min(P, (b + 1) * BOX_SIZE); i++)
		{
			if (i == idx)
				continue;
			updateKBest<3>(point, points[indices[i]], best);
		}
	}
	dists[indices[idx]] = (best[0] + best[1] + best[2]) / 3.0f;
}

__global__ void boxMeanDistb(uint32_t P1, float3* points1, uint32_t* indices1,  uint32_t P2, float3* points2, uint32_t* indices2, MinMax* boxes, float3* dists)
{

	int idx = cg::this_grid().thread_rank();
	if (idx >= P1)
		return;

	float3 point = points1[idx];
	float best[5] = { FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX };
	float3 bestxyz[5] = {points1[idx], points1[idx], points1[idx], points1[idx], points1[idx]};

	for (int i = max(0, indices1[idx] - 3); i <= min(P2 - 1, indices1[idx] + 3); i++)
	{
		// if (i == idx)
		// 	continue;
		updateKBest<5>(point, points2[indices2[i]], best);
	}

	float reject = best[4];
	best[0] = FLT_MAX;
	best[1] = FLT_MAX;
	best[2] = FLT_MAX;
	best[3] = FLT_MAX;
	best[4] = FLT_MAX;

	for (int b = 0; b < (P2 + BOX_SIZE - 1) / BOX_SIZE; b++)
	{
		MinMax box = boxes[b];
		float dist = distBoxPoint(box, point);
		if (dist > reject || dist > best[4])
			continue;

		for (int i = b * BOX_SIZE; i < min(P2, (b + 1) * BOX_SIZE); i++)
		{
			// if (i == idx)
			// 	continue;
			updateKBestv<5>(point, points2[indices2[i]], best, bestxyz);
		}
	}
	//dists[idx] = (bestxyz[0] + bestxyz[1] + bestxyz[2]) / 3.0f;
	dists[idx] = make_float3((bestxyz[0].x + bestxyz[1].x + bestxyz[2].x + bestxyz[3].x + bestxyz[4].x) / 5.0f, (bestxyz[0].y + bestxyz[1].y + bestxyz[2].y + bestxyz[3].y + bestxyz[4].y) / 5.0f,  (bestxyz[0].z + bestxyz[1].z + bestxyz[2].z + bestxyz[3].z + bestxyz[4].z) / 5.0f);
}

void SimpleKNN::knn(int P, float3* points, float* meanDists)
{
	float3* result;
	cudaMalloc(&result, sizeof(float3));
	size_t temp_storage_bytes;

	float3 init = { 0, 0, 0 }, minn, maxx;

	cub::DeviceReduce::Reduce(nullptr, temp_storage_bytes, points, result, P, CustomMin(), init);
	thrust::device_vector<char> temp_storage(temp_storage_bytes);

	cub::DeviceReduce::Reduce(temp_storage.data().get(), temp_storage_bytes, points, result, P, CustomMin(), init);
	cudaMemcpy(&minn, result, sizeof(float3), cudaMemcpyDeviceToHost);

	cub::DeviceReduce::Reduce(temp_storage.data().get(), temp_storage_bytes, points, result, P, CustomMax(), init);
	cudaMemcpy(&maxx, result, sizeof(float3), cudaMemcpyDeviceToHost);

	thrust::device_vector<uint32_t> morton(P);
	thrust::device_vector<uint32_t> morton_sorted(P);
	coord2Morton << <(P + 255) / 256, 256 >> > (P, points, minn, maxx, morton.data().get());

	thrust::device_vector<uint32_t> indices(P);
	thrust::sequence(indices.begin(), indices.end());
	thrust::device_vector<uint32_t> indices_sorted(P);

	cub::DeviceRadixSort::SortPairs(nullptr, temp_storage_bytes, morton.data().get(), morton_sorted.data().get(), indices.data().get(), indices_sorted.data().get(), P);
	temp_storage.resize(temp_storage_bytes);

	cub::DeviceRadixSort::SortPairs(temp_storage.data().get(), temp_storage_bytes, morton.data().get(), morton_sorted.data().get(), indices.data().get(), indices_sorted.data().get(), P);

	uint32_t num_boxes = (P + BOX_SIZE - 1) / BOX_SIZE;
	thrust::device_vector<MinMax> boxes(num_boxes);
	boxMinMax << <num_boxes, BOX_SIZE >> > (P, points, indices_sorted.data().get(), boxes.data().get());
	boxMeanDist << <num_boxes, BOX_SIZE >> > (P, points, indices_sorted.data().get(), boxes.data().get(), meanDists);

	cudaFree(result);
}

void SimpleKNN::knnb(int P1, float3* points1, int P2, float3* points2, float3* meanDists)
{
	float3* result;
	// float3* result2;
	cudaMalloc(&result, sizeof(float3));
	//cudaMalloc(&result2, sizeof(float3));
	size_t temp_storage_bytes;

	float3 init = { 0, 0, 0 }, minn, maxx;

	// allocate P1+P2 float3s
	thrust::device_vector<float3> all(P1 + P2);

	// copy both arrays into it
	thrust::copy(points1, points1 + P1, all.begin());
	thrust::copy(points2, points2 + P2, all.begin() + P1);


	cub::DeviceReduce::Reduce(nullptr, temp_storage_bytes, all.data().get(), result, P1 + P2, CustomMin(), init);
	thrust::device_vector<char> temp_storage(temp_storage_bytes);

	cub::DeviceReduce::Reduce(temp_storage.data().get(), temp_storage_bytes, all.data().get(), result, P1 + P2, CustomMin(), init);
	cudaMemcpy(&minn, result, sizeof(float3), cudaMemcpyDeviceToHost);

	cub::DeviceReduce::Reduce(temp_storage.data().get(), temp_storage_bytes, all.data().get(), result, P1 + P2, CustomMax(), init);
	cudaMemcpy(&maxx, result, sizeof(float3), cudaMemcpyDeviceToHost);

	// cub::DeviceReduce::Reduce(nullptr, temp_storage_bytes, points1, result1, P1, CustomMin(), init);
	// thrust::device_vector<char> temp_storage(temp_storage_bytes);

	// cub::DeviceReduce::Reduce(temp_storage.data().get(), temp_storage_bytes, points1, result1, P1, CustomMin(), init);
	// cudaMemcpy(&minn1, result1, sizeof(float3), cudaMemcpyDeviceToHost);

	// cub::DeviceReduce::Reduce(temp_storage.data().get(), temp_storage_bytes, points1, result1, P1, CustomMax(), init);
	// cudaMemcpy(&maxx1, result1, sizeof(float3), cudaMemcpyDeviceToHost);

	thrust::device_vector<uint32_t> morton2(P2);
	thrust::device_vector<uint32_t> morton1(P1);
	thrust::device_vector<uint32_t> morton2_sorted(P2);
	coord2Morton << <(P2 + 255) / 256, 256 >> > (P2, points2, minn, maxx, morton2.data().get());
	coord2Morton << <(P1 + 255) / 256, 256 >> > (P1, points1, minn, maxx, morton1.data().get());

	thrust::device_vector<uint32_t> indices2(P2);
	thrust::sequence(indices2.begin(), indices2.end());
	thrust::device_vector<uint32_t> indices2_sorted(P2);
	thrust::device_vector<uint32_t> indices1_sorted(P1);

	cub::DeviceRadixSort::SortPairs(nullptr, temp_storage_bytes, morton2.data().get(), morton2_sorted.data().get(), indices2.data().get(), indices2_sorted.data().get(), P2);
	temp_storage.resize(temp_storage_bytes);

	cub::DeviceRadixSort::SortPairs(temp_storage.data().get(), temp_storage_bytes, morton2.data().get(), morton2_sorted.data().get(), indices2.data().get(), indices2_sorted.data().get(), P2);

	uint32_t num_boxes = (P2 + BOX_SIZE - 1) / BOX_SIZE;
	thrust::device_vector<MinMax> boxes(num_boxes);
	boxMinMax << <num_boxes, BOX_SIZE >> > (P2, points2, indices2_sorted.data().get(), boxes.data().get());
	thrust::lower_bound(morton2_sorted.begin(), morton2_sorted.end(),morton1.begin(), morton1.end(), indices1_sorted.begin());
	//boxMeanDist << <num_boxes, BOX_SIZE >> > (P, points, indices_sorted.data().get(), boxes.data().get(), meanDists);
	boxMeanDistb << <num_boxes, BOX_SIZE >> > (P1, points1, indices1_sorted.data().get(), P2, points2, indices2_sorted.data().get(), boxes.data().get(), meanDists);
	cudaFree(result);
}