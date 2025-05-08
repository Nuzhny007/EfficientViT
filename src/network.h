#pragma once

#include "TorchHeader.h"

#include <memory>
#include <string>
#include <unordered_map>
#include <functional>
#include <optional>


///Функция для вычисления padding'а, если передан int
inline int get_same_padding(int kernel_size)
{
	assert(kernel_size % 2 > 0 && "kernel size should be odd number");
	return kernel_size / 2;
}

///Функция для вычисления padding'а, если передан std::vector<int>
inline std::vector<int> get_same_padding(const std::vector<int> &kernel_sizes)
{
	std::vector<int> result;
	for (const auto &ks : kernel_sizes)
	{
		assert(ks % 2 > 0 && "kernel size should be odd number");
		result.push_back(ks / 2);
	}
	return result;
}
