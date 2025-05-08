#pragma once

#include "TorchHeader.h"

#include "act.h"
#include "norm.h"
#include "network.h"

#include <memory>
#include <string>
#include <unordered_map>
#include <functional>
#include <optional>

/********************************************************************************/
//Basic layers
/********************************************************************************/

///
class ConvLayerImpl : public torch::nn::Module {
private:
	torch::nn::Dropout2d DropoutLayer{nullptr};
	torch::nn::Conv2d Conv{nullptr};
	std::shared_ptr<torch::nn::Module> NormLayer{nullptr},
		Act{nullptr};
	std::string NormLayerName,
		ActName;
public:
	ConvLayerImpl(
		const int in_channels,
		const int out_channels,
		const int kernel_size = 3,
		const int stride = 1,
		const int dilation = 1,
		const int groups = 1,
		const bool use_bias = false,
		const float dropout = 0,
		const std::string &norm = "bn2d",
		const std::string &act_func = "relu"
	) : NormLayerName(norm), ActName(act_func)
	{
		int padding = get_same_padding(kernel_size) * dilation;

		if (dropout > 0)
		{
			DropoutLayer = torch::nn::Dropout2d(torch::nn::Dropout2dOptions().p(dropout).inplace(false));
			register_module("dropout", DropoutLayer);
		}
		Conv = torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, { kernel_size, kernel_size}).padding(padding).stride({ stride, stride }).dilation({ dilation, dilation }).groups(groups).bias(use_bias));
		NormLayer = build_norm(norm, out_channels);
		Act = build_act(act_func);
		
		register_module("conv", Conv);
		if (NormLayer != nullptr)
			register_module("norm", NormLayer);
		if (Act != nullptr)
			register_module("act", Act);
	}
	virtual ~ConvLayerImpl() = default;

	torch::Tensor forward(torch::Tensor x)
	{
		if (!DropoutLayer.is_empty())
			x = DropoutLayer->forward(x);
		x = Conv->forward(x);
		if (NormLayer != nullptr)
			x = forward_norm(NormLayer, x, NormLayerName);
		if (Act != nullptr)
			x = forward_act(Act, x, ActName);
		return x;
	}
};
TORCH_MODULE(ConvLayer);


///
class UpSampleLayerImpl : public torch::nn::Module {
private:
	std::optional<std::vector<int64_t>> Size{ std::nullopt };
	std::optional<std::vector<double>> Factor{std::nullopt};
	torch::nn::functional::InterpolateFuncOptions::mode_t Mode;
	bool AlignCorners;

	//Преобразует размер в вектор из 2 элементов
	std::vector<int64_t> parse_size(const std::vector<int64_t> &size)
	{
		if (size.size() == 1)
		{
			return { size[0], size[0] };
		} else if (size.size() == 2) {
			return size;
		} else {
			throw std::runtime_error("Size must be a 1 or 2-element vector");
		}
	}

	//Проверяет, совпадает ли текущий размер с целевым
	bool check_current_size(const torch::Tensor& x, const std::vector<int64_t>& target_size) {
		auto current_size = x.sizes();
		return (current_size[current_size.size() - 2] == target_size[0]) &&
			(current_size[current_size.size() - 1] == target_size[1]);
	}

	//Функция интерполяции
	torch::Tensor resize(
		const torch::Tensor& x,
		std::optional<std::vector<int64_t>> size,
		std::optional<std::vector<double>> factor,
		const torch::nn::functional::InterpolateFuncOptions::mode_t interpolate_mode,
		bool align_corners
	) {
		auto options = torch::nn::functional::InterpolateFuncOptions()
			.mode(interpolate_mode)
			.align_corners(align_corners)
			.size(size)
			.scale_factor(factor);
		return torch::nn::functional::interpolate(x, options);
	}

public:
	UpSampleLayerImpl(
		torch::nn::functional::InterpolateFuncOptions::mode_t mode = torch::kBicubic,
		std::optional<std::vector<int64_t>> size = std::nullopt,
		int64_t factor = 2,
		bool align_corners = false
	) : Mode(mode), AlignCorners(align_corners)
	{
		if (size.has_value())
		{
			Size = parse_size(size.value());
		} else {
			Factor = std::vector<double>{ (double)factor };
		}
	}

	torch::Tensor forward(torch::Tensor x)
	{
		//Проверка, нужно ли выполнять апсемплинг
		if ((Size.has_value() && check_current_size(x, Size.value())) || (Factor.has_value() && Factor.value()[0] == 1 && Factor.value()[1] == 1))
			return x;

		//Конвертация типов, если необходимо
		if (x.dtype() == torch::kFloat16 || x.dtype() == torch::kBFloat16)
			x = x.to(torch::kFloat32);

		//Применение интерполяции
		return resize(x, Size, Factor, Mode, AlignCorners);
	}
};
TORCH_MODULE(UpSampleLayer);
///!!!Можно заменить на torch::nn::Upsample model(torch::nn::UpsampleOptions().scale_factor(std::vector<double>({2})).mode(torch::kBicubic).align_corners(false));



///
class ConvPixelUnshuffleDownSampleLayerImpl : public torch::nn::Module {
private:
	int Factor;
	ConvLayer Conv {nullptr};
public:
	ConvPixelUnshuffleDownSampleLayerImpl(
		const int in_channels,
		const int out_channels,
		const int kernel_size,
		const int factor
	) : Factor(factor)
	{
		auto out_ratio = factor * factor;
		if (out_channels % out_ratio != 0)
			throw std::invalid_argument("out_channels must be divisible by the square of factor");

		Conv = register_module("conv", ConvLayer(
			in_channels,
			out_channels / out_ratio,
			kernel_size,
			1,	//stride
			1,	//dilation
			1,	//groups
			true, // use_bias
			0,	// dropout,
			"", // norm
			""	// act_func
		));
	}
	virtual ~ConvPixelUnshuffleDownSampleLayerImpl() = default;

	torch::Tensor forward(torch::Tensor x)// const 
	{
		x = Conv->forward(x);
		torch::nn::functional::PixelUnshuffleFuncOptions options(/*downscale_factor=*/Factor);
		x = torch::nn::functional::pixel_unshuffle(x, Factor);
		return x;
	}
};
TORCH_MODULE(ConvPixelUnshuffleDownSampleLayer);


///
class PixelUnshuffleChannelAveragingDownSampleLayerImpl : public torch::nn::Module {
private:
	int InChannels;
	int OutChannels;
	int Factor;
	int GroupSize;
public:
	PixelUnshuffleChannelAveragingDownSampleLayerImpl(const int in_channels, const int out_channels, const int factor)
		: InChannels(in_channels), OutChannels(out_channels), Factor(factor)
	{
		assert(in_channels * factor * factor % out_channels == 0);		//!!!
		GroupSize = (in_channels * factor * factor) / out_channels;
	}
	virtual ~PixelUnshuffleChannelAveragingDownSampleLayerImpl() = default;

	torch::Tensor forward(torch::Tensor x)
	{
		torch::nn::functional::PixelUnshuffleFuncOptions options(/*downscale_factor=*/Factor);
		x = torch::nn::functional::pixel_unshuffle(x, options);
		// Reshape and compute mean
		auto sizes = x.sizes();
		int B = sizes[0];
		int C = sizes[1];
		int H = sizes[2];
		int W = sizes[3];

		x = x.view({B, OutChannels, GroupSize, H, W});
		x = x.mean(/*dim=*/2);

		return x;
	}
};
TORCH_MODULE(PixelUnshuffleChannelAveragingDownSampleLayer);


///
class ConvPixelShuffleUpSampleLayerImpl : public torch::nn::Module {
private:
	int Factor;
	int OutRatio;
	ConvLayer Conv{nullptr};
public:
	ConvPixelShuffleUpSampleLayerImpl(const int in_channels, const int out_channels, const int kernel_size, const int factor)
		: Factor(factor)
	{
		OutRatio = factor * factor;
		
		Conv = register_module("conv", ConvLayer(
			in_channels,
			out_channels * OutRatio,
			kernel_size,
			1,	//stride
			1,	//dilation
			1,	//groups
			true, // use_bias
			0,	// dropout,
			"", // norm
			""	// act_func
		));
	}
	virtual ~ConvPixelShuffleUpSampleLayerImpl() = default;

	torch::Tensor forward(torch::Tensor x)
	{
		x = Conv->forward(x);
		torch::nn::functional::PixelShuffleFuncOptions options(Factor);
		x = torch::nn::functional::pixel_shuffle(x, options);
		return x;
	}
};
TORCH_MODULE(ConvPixelShuffleUpSampleLayer);


///
class InterpolateConvUpSampleLayerImpl : public torch::nn::Module {
private:
	int Factor;
	torch::nn::functional::InterpolateFuncOptions::mode_t Mode;
	ConvLayer Conv{nullptr};
public:
	InterpolateConvUpSampleLayerImpl(
		const int in_channels,
		const int out_channels,
		const int kernel_size,
		const int factor,
		torch::nn::functional::InterpolateFuncOptions::mode_t mode = torch::kNearest
	) : Factor(factor), Mode(mode)
	{
		Conv = register_module("conv_layer", ConvLayer(
			in_channels,
			out_channels,
			kernel_size,
			1,	//stride
			1,	//dilation
			1,	//groups
			true, // use_bias
			0,	// dropout,
			"", // norm
			""	// act_func
		));
	}
	virtual ~InterpolateConvUpSampleLayerImpl() = default;

	torch::Tensor forward(torch::Tensor x)
	{
		namespace F = torch::nn::functional;
		F::InterpolateFuncOptions options;
		options.mode(Mode).scale_factor(std::vector<double>{(double)Factor, (double)Factor});
		x = F::interpolate(x, options);
		x = Conv->forward(x);
		return x;
	}
};
TORCH_MODULE(InterpolateConvUpSampleLayer);


///
class ChannelDuplicatingPixelUnshuffleUpSampleLayerImpl : public torch::nn::Module {
private:
	int InChannels,
		OutChannels,
		Factor,
		Repeats;
public:
	ChannelDuplicatingPixelUnshuffleUpSampleLayerImpl(const int in_channels, const int out_channels, const int factor)
		: InChannels(in_channels), OutChannels(out_channels), Factor(factor)
	{
		assert(out_channels * factor * factor % in_channels == 0);
		Repeats = (out_channels * factor * factor) / in_channels;
	}
	virtual ~ChannelDuplicatingPixelUnshuffleUpSampleLayerImpl() = default;

	torch::Tensor forward(torch::Tensor x)
	{
		x = x.repeat_interleave(Repeats, /*dim=*/1);		// Повторение каналов
		torch::nn::functional::PixelShuffleFuncOptions options(Factor);
		x = torch::nn::functional::pixel_shuffle(x, options);
		return x;
	}
};
TORCH_MODULE(ChannelDuplicatingPixelUnshuffleUpSampleLayer);


///
class LinearLayerImpl : public torch::nn::Module {
private:
	torch::nn::Dropout DropoutLayer{nullptr};
	torch::nn::Linear LinearLayer{nullptr};
	std::shared_ptr<torch::nn::Module> NormLayer{nullptr},
		Act{nullptr};
	std::string NormLayerName,
		ActName;
public:

	explicit LinearLayerImpl(
		const int in_features,
		const int out_features,
		const bool use_bias = true,
		const double &dropout = 0.,
		const std::string &norm = "",
		const std::string &act_func = ""
	) : NormLayerName(norm), ActName(act_func)
	{
		if (dropout > 0)
		{
			DropoutLayer = torch::nn::Dropout(torch::nn::DropoutOptions().p(dropout).inplace(false));
			register_module("dropout", DropoutLayer);
		}

		LinearLayer = register_module("linear", torch::nn::Linear(torch::nn::LinearOptions(in_features, out_features).bias(use_bias)));

		if (!norm.empty())
		{
			NormLayer = build_norm(norm, out_features);
		}
		if (!act_func.empty())
		{
			Act = build_act(act_func);
		}
		if (NormLayer != nullptr)
			register_module("norm", NormLayer);
		if (Act != nullptr)
			register_module("act", Act);
	}
	virtual ~LinearLayerImpl() = default;

	torch::Tensor forward(torch::Tensor x)
	{
		//Попытка сжать тензор до 2-х измерений
		if (x.dim() > 2)
		{
			x = x.flatten(1); //Сжимаем начиная с первого измерения
		}
		if (!DropoutLayer.is_empty())
			x = DropoutLayer->forward(x);
		x = LinearLayer->forward(x);
		if (NormLayer != nullptr)
			x = forward_norm(NormLayer, x, NormLayerName);
		if (Act != nullptr)
			x = forward_act(Act, x, ActName);
		return x;
	}
};
TORCH_MODULE(LinearLayer);


///
class IdentityLayerImpl : public torch::nn::Module {
public:
	virtual ~IdentityLayerImpl() = default;
	torch::Tensor forward(torch::Tensor x)
	{
		return x;
	}
};
TORCH_MODULE(IdentityLayer);


/********************************************************************************/
//Basic blocks
/********************************************************************************/


///Функция для преобразования значений в пару, аналогично val2tuple из Python
template<typename T>
std::pair<T, T> val2tuple(T value)
{
	return {value, value};
}

template<typename T>
std::tuple<T, T, T> val3tuple(T value)
{
	return {value, value, value};
}


///
class DSConvImpl : public torch::nn::Module {
private:
	ConvLayer DepthConv {nullptr},
		PointConv {nullptr};
public:
	explicit DSConvImpl(
		const int in_channels,
		const int out_channels,
		const int kernel_size = 3,
		const int stride = 1,
		const bool use_bias = false,
		const std::pair<std::string, std::string> &norm = { "bn2d", "bn2d" },
		const std::pair<std::string, std::string> &act_func = { "relu6", "" }
	){
		// Преобразование параметров в пары
		auto use_bias_pair = val2tuple<bool>(use_bias);

		DepthConv = register_module("depth_conv", ConvLayer(
			in_channels,
			in_channels,
			kernel_size,
			stride,
			1/*dilation*/,
			in_channels, // groups = in_channels для глубинной свертки
			use_bias_pair.first,
			0/*dropuot*/,
			norm.first,
			act_func.first
		));

		PointConv = register_module("point_conv", ConvLayer(
			in_channels,
			out_channels,
			1/*kernel_size*/,
			1/*stride*/,
			1/*dilation*/,
			1, // groups = 1 для точечной свертки
			use_bias_pair.second,
			0/*dropuot*/,
			norm.second,
			act_func.second
		));
	}
	virtual ~DSConvImpl() = default;

	torch::Tensor forward(torch::Tensor x)
	{
		x = DepthConv->forward(x);
		x = PointConv->forward(x);
		return x;
	}
};
TORCH_MODULE(DSConv);


///
class MBConvImpl : public torch::nn::Module {
private:
	ConvLayer InvertedConv {nullptr},
		DepthConv {nullptr},
		PointConv {nullptr};
	int MidChannels;
public:
	explicit MBConvImpl(
		const int in_channels,
		const int out_channels,
		const int kernel_size = 3,
		const int stride = 1,
		const c10::optional<int64_t> mid_channels = c10::nullopt,
		const float &expand_ratio = 6.0f,
		const std::tuple <bool, bool, bool> use_bias = {false, false, false},
		const std::tuple<std::string, std::string, std::string> &norm = {"bn2d", "bn2d", "bn2d"},
		const std::tuple<std::string, std::string, std::string> &act_func = {"relu6", "relu6", ""}
	){
		MidChannels = mid_channels.has_value() ? mid_channels.value() : static_cast<int64_t>(std::round(in_channels * expand_ratio));

		InvertedConv = register_module("inverted_conv", ConvLayer(
			in_channels,
			MidChannels,
			1/*kernel_size*/,
			1/*stride*/,
			1/*dilation*/,
			1, // groups = 1
			std::get<0>(use_bias),
			0/*dropuot*/,
			std::get<0>(norm),
			std::get<0>(act_func)
		));

		DepthConv = register_module("depth_conv", ConvLayer(
			MidChannels,
			MidChannels,
			kernel_size,
			stride,
			1/*dilation*/,
			MidChannels, // groups = mid_channels для глубинной свертки
			std::get<1>(use_bias),
			0/*dropuot*/,
			std::get<1>(norm),
			std::get<1>(act_func)
		));

		PointConv = register_module("point_conv", ConvLayer(
			MidChannels,
			out_channels,
			1/*kernel_size*/,
			1/*stride*/,
			1/*dilation*/,
			1, // groups = 1
			std::get<2>(use_bias),
			0/*dropuot*/,
			std::get<2>(norm),
			std::get<2>(act_func)
		));
	}
	virtual ~MBConvImpl() = default;

	torch::Tensor forward(torch::Tensor x)
	{
		x = InvertedConv->forward(x);
		x = DepthConv->forward(x);
		x = PointConv->forward(x);
		return x;
	}
};
TORCH_MODULE(MBConv);


///
class FusedMBConvImpl : public torch::nn::Module {
private:
	ConvLayer SpatialConv {nullptr};
	ConvLayer PointConv {nullptr};
	int MidChannels;
public:
	explicit FusedMBConvImpl(
		const int in_channels,
		const int out_channels,
		const int kernel_size = 3,
		const int stride = 1,
		const c10::optional<int> mid_channels = c10::nullopt,
		const double &expand_ratio = 6.0,
		const int groups = 1,
		const bool use_bias = false,
		const std::pair<std::string, std::string> &norm = {"bn2d", "bn2d"},
		const std::pair<std::string, std::string> &act_func = {"relu6", ""}
	){
		// Преобразование параметров в пары
		auto use_bias_tuple = val2tuple<bool>(use_bias);

		MidChannels = mid_channels.has_value() ? mid_channels.value() : static_cast<int>(std::round(in_channels * expand_ratio));

		SpatialConv = register_module("spatial_conv", ConvLayer(
			in_channels,
			MidChannels,
			kernel_size,
			stride,
			1/*dilation*/,
			groups, // groups для пространственной свертки
			use_bias_tuple.first,
			0/*dropuot*/,
			std::get<0>(norm),
			std::get<0>(act_func)
		));

		PointConv = register_module("point_conv", ConvLayer(
			MidChannels,
			out_channels,
			1/*kernel_size*/,
			1, // ядро и шаг равны 1 для точечной свертки
			1/*dilation*/,
			1, // groups = 1
			use_bias_tuple.second,
			0/*dropuot*/,
			std::get<1>(norm),
			std::get<1>(act_func)
		));
	}
	virtual ~FusedMBConvImpl() = default;

	torch::Tensor forward(torch::Tensor x) 
	{
		x = SpatialConv->forward(x);
		x = PointConv->forward(x);
		return x;
	}
};
TORCH_MODULE(FusedMBConv);


///
class GLUMBConvImpl : public torch::nn::Module {
private:
	int MidChannels;
	int InvertedConvOutChannels;
	std::shared_ptr<torch::nn::Module> GluAct{nullptr};
	std::string GluActName;
	ConvLayer InvertedConv {nullptr},
		DepthConv {nullptr},
		PointConv {nullptr};
public:
	explicit GLUMBConvImpl(
		const int in_channels,
		const int out_channels,
		const int kernel_size = 3,
		const int stride = 1,
		const c10::optional<int> mid_channels = c10::nullopt,
		const double &expand_ratio = 6.0,
		const std::tuple <bool, bool, bool> use_bias = { false, false, false },
		const std::tuple<std::string, std::string, std::string> &norm = {"", "", "ln2d"},
		const std::tuple<std::string, std::string, std::string> &act_func = {"silu", "silu", ""})
	{
		MidChannels = mid_channels.has_value() ? mid_channels.value() : static_cast<int>(std::round(in_channels * expand_ratio));

		InvertedConvOutChannels = MidChannels * 2;

		GluActName = std::get<1>(act_func);
		GluAct = build_act(GluActName /* inplace=false по умолчанию */);
		if (GluAct != nullptr)
			register_module("glu_act", GluAct);

		InvertedConv = register_module("inverted_conv", ConvLayer(
			in_channels,
			InvertedConvOutChannels,
			1, // ядро и шаг равны 1 для инвертированной свертки
			1, /*stride*/
			1, /*dilation*/
			1, /*groups*/
			std::get<0>(use_bias),
			0/*dropuot*/,
			std::get<0>(norm),
			std::get<0>(act_func)
		));

		DepthConv = register_module("depth_conv", ConvLayer(
			InvertedConvOutChannels,
			InvertedConvOutChannels,
			kernel_size,
			stride,
			1, /*dilation*/
			InvertedConvOutChannels, /*groups*/
			std::get<1>(use_bias),
			0/*dropuot*/,
			std::get<1>(norm),
			"" // нет активационной функции
		)); 

		PointConv = register_module("point_conv", ConvLayer(
			MidChannels,
			out_channels,
			1, // ядро и шаг равны 1 для точечной свертки
			1, /*stride*/
			1, /*dilation*/
			1, /*groups*/
			std::get<2>(use_bias),
			0/*dropuot*/,
			std::get<2>(norm),
			std::get<2>(act_func)
		));
	}
	virtual ~GLUMBConvImpl() = default;

	torch::Tensor forward(torch::Tensor x)
	{
		x = InvertedConv->forward(x);
		x = DepthConv->forward(x);
		auto chunks = torch::chunk(x, 2, /*dim=*/1);		//Разделение тензора на два части по размеру канала
		auto gate = chunks[1];
		if (GluAct != nullptr)
			gate = forward_act(GluAct, gate, GluActName);
		x = chunks[0] * gate;
		x = PointConv->forward(x);
		return x;
	}
};
TORCH_MODULE(GLUMBConv);


///
class ResBlockImpl : public torch::nn::Module {
private:
	int MidChannels;
	ConvLayer Conv1 {nullptr},
		Conv2 {nullptr};
public:
	explicit ResBlockImpl(
		const int in_channels,
		const int out_channels,
		const int kernel_size = 3,
		const int stride = 1,
		const c10::optional<int64_t> mid_channels = c10::nullopt,
		const double &expand_ratio = 1.0,
		const std::pair <bool, bool> use_bias = { false, false },
		const std::pair<std::string, std::string> &norm = {"bn2d", "bn2d"},
		const std::pair<std::string, std::string> &act_func = {"relu6", ""}
	){
		MidChannels = mid_channels.has_value() ? mid_channels.value() : static_cast<int>(std::round(in_channels * expand_ratio));

		Conv1 = register_module("conv1", ConvLayer(
			in_channels,
			MidChannels,
			kernel_size,
			stride,
			1/*dilation*/,
			1/*groups*/,
			use_bias.first,
			0/*dropuot*/,
			std::get<0>(norm),
			std::get<0>(act_func)
		));

		Conv2 = register_module("conv2", ConvLayer(
			MidChannels,
			out_channels,
			kernel_size,
			1/*stride*/,
			1/*dilation*/,
			1/*groups*/,
			use_bias.second,
			0/*dropuot*/,
			std::get<1>(norm),
			std::get<1>(act_func)
		));
	}
	virtual ~ResBlockImpl() = default;

	torch::Tensor forward(torch::Tensor x)
	{
		x = Conv1->forward(x);
		x = Conv2->forward(x);
		return x;
	}
};
TORCH_MODULE(ResBlock);


///
class LiteMLAImpl : public torch::nn::Module {
private:
	int Heads;
	int TotalDim;
	int Dim;
	ConvLayer QKV {nullptr};
	std::vector<torch::nn::Sequential> Aggreg;
	std::shared_ptr<torch::nn::Module> KernelFunc {nullptr};
	std::string KernelFuncName;
	ConvLayer Proj {nullptr};
	float Eps;
public:
	explicit LiteMLAImpl(
		const int in_channels,
		const int out_channels,
		const c10::optional<int> heads = c10::nullopt,
		const float heads_ratio = 1.0,
		const int dim = 8,
		const bool use_bias = false,
		const std::pair<std::string, std::string> &norm = { "", "bn2d" },
		const std::pair<std::string, std::string> &act_func = {"", ""},
		const std::string &kernel_func = "relu",
		const std::vector<int> &scales = {5},
		float eps = 1.0e-15) : KernelFuncName(kernel_func), Eps(eps)
	{
		Heads = heads.has_value() ? heads.value() : static_cast<int64_t>(in_channels * heads_ratio / dim );
		TotalDim = Heads * dim;
		auto use_bias_tuple = val2tuple(use_bias);
		Dim = dim;

		QKV = register_module("qkv", ConvLayer(
			in_channels,
			3 * TotalDim,
			1, //kernel_size
			1, //stride
			1, //dilation
			1, //groups
			use_bias_tuple.first,
			0, //dropout
			norm.first,
			act_func.first
		));

		for (auto scale : scales)
		{
			auto seq = register_module("seq" + std::to_string(Aggreg.size()), torch::nn::Sequential(
				torch::nn::Conv2d(torch::nn::Conv2dOptions(3 * TotalDim, 3 * TotalDim, scale)
					.padding(get_same_padding(scale))
					.groups(3 * TotalDim)
					.bias(use_bias_tuple.first)
				),
				torch::nn::Conv2d(torch::nn::Conv2dOptions(3 * TotalDim, 3 * TotalDim, 1)
					.groups(3 * Heads)
					.bias(use_bias_tuple.first)
				)
			));
			Aggreg.push_back(seq);
		}

		KernelFunc = build_act(kernel_func /*,inplace = false по умолчанию*/);
		if (KernelFunc != nullptr)
			register_module("kernel_func", KernelFunc);

		Proj = register_module("proj", ConvLayer(
			TotalDim * (1 + scales.size()),
			out_channels,
			1, // kernel_size
			1, //stride
			1, //dilation
			1, //groups
			use_bias_tuple.second,
			0, //dropout
			norm.second,
			act_func.second
		));
	}
	virtual ~LiteMLAImpl() = default;

	torch::Tensor relu_linear_att(torch::Tensor qkv)
	{
		auto B = qkv.size(0);
		auto H = qkv.size(2);
		auto W = qkv.size(3);

		if (qkv.dtype() == torch::kFloat16)
			qkv = qkv.to(torch::kFloat);

		qkv = torch::reshape(qkv, {B, -1, 3 * Dim, H * W});
		auto q = qkv.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(0, Dim)});
		auto k = qkv.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(Dim, 2 * Dim)});
		auto v = qkv.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(2 * Dim, torch::indexing::None)});

		//Lightweight linear attention
		q = forward_act(KernelFunc, q, KernelFuncName);
		k = forward_act(KernelFunc, k, KernelFuncName);

		//Linear matmul
		auto trans_k = k.transpose(/*dim0=*/-1, /*dim1=*/-2);

		v = torch::nn::functional::pad(v, torch::nn::functional::PadFuncOptions({0, 0, 0, 1}).mode(torch::kConstant).value(1));
		auto vk = torch::matmul(v, trans_k);
		auto out = torch::matmul(vk, q);
		if (out.dtype() == torch::kBFloat16)
		{
			out = out.to(torch::kFloat);
		}
		out = out.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(torch::indexing::None, -1)}) / 
			(out.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(-1, torch::indexing::None)}) + Eps);

		return torch::reshape(out, {B, -1, H, W});
	}

	torch::Tensor relu_quadratic_att(torch::Tensor qkv)
	{
		auto B = qkv.size(0);
		auto H = qkv.size(2);
		auto W = qkv.size(3);

		//qkv = qkv.to(torch::kFloat);

		qkv = torch::reshape(qkv, {B, -1, 3 * Dim, H * W});
		auto q = qkv.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(0, Dim)});
		auto k = qkv.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(Dim, 2 * Dim)});
		auto v = qkv.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(2 * Dim, torch::indexing::None)});

		q = forward_act(KernelFunc, q, KernelFuncName);
		k = forward_act(KernelFunc, k, KernelFuncName);

		auto att_map = torch::matmul(k.transpose(/*dim0=*/-1, /*dim1=*/-2), q); // b h n n
		auto original_dtype = att_map.dtype();
		if (original_dtype == torch::kFloat16 || original_dtype == torch::kBFloat16)
		{
			att_map = att_map.to(torch::kFloat);
		}
		att_map = att_map / (att_map.sum(/*dim=*/2, /*keepdim=*/true) + Eps);
		att_map = att_map.to(original_dtype);
		auto out = torch::matmul(v, att_map); // b h d n

		return torch::reshape(out, {B, -1, H, W});
	}

	torch::Tensor forward(torch::Tensor x)
	{
		//x = x.to(torch::kFloat);
		//Generate multi-scale q, k, v
		auto qkv = QKV->forward(x);
		std::vector<torch::Tensor> multi_scale_qkv = {qkv};
		for (auto &op : Aggreg)
		{
			multi_scale_qkv.push_back(op->forward(qkv));
		}
		qkv = torch::cat(multi_scale_qkv, /*dim=*/1);

		auto H = qkv.size(2);
		auto W = qkv.size(3);
		torch::Tensor out;
		if (H * W > Dim)
		{
			out = relu_linear_att(qkv).to(qkv.dtype());
		} else {
			out = relu_quadratic_att(qkv);
		}
		out = Proj->forward(out);

		return out;
	}
};
TORCH_MODULE(LiteMLA);


/********************************************************************************/
//Functional blocks
/********************************************************************************/

///
template <typename TMain, typename TShortcut>
class ResidualBlockImpl : public torch::nn::Module {
private:
	TMain Main {nullptr};
	TShortcut Shortcut {nullptr};
	std::shared_ptr<torch::nn::Module> PostAct {nullptr},
		PreNorm {nullptr};
	std::string PostActName,
		PreNormName;
public:
	explicit ResidualBlockImpl(
		TMain main_block,
		TShortcut shortcut,
		const std::string &post_act_name = "",
		const std::string &pre_norm_name = "",
		std::shared_ptr<torch::nn::Module> pre_norm = nullptr		///Можно получать текстовое описание const std::string & norm = "bn2d" по аналогии с EfficientViTBlock
	) : PostActName(post_act_name), PreNormName(pre_norm_name) {
		if (!main_block.is_empty())
			Main = register_module("main", main_block);
		if (!shortcut.is_empty())
			Shortcut = register_module("shortcut", shortcut);
		if (!post_act_name.empty())
		{
			PostAct = build_act(post_act_name);
			if (PostAct != nullptr)
				PostAct = register_module("post_act", PostAct);
		}
		if (pre_norm != nullptr)
			PreNorm = register_module("pre_norm", pre_norm);
	}
	virtual ~ResidualBlockImpl() = default;

	torch::Tensor forward_main(torch::Tensor x)
	{
		if (PreNorm == nullptr)
		{
			return Main->forward(x);
		} else {
			return Main->forward(forward_norm(PreNorm, x, PreNormName));
		}
	}

	torch::Tensor forward(torch::Tensor x)
	{
		torch::Tensor res;
		if (Main.is_empty())
		{
			res = x;
		} else {
			if (Shortcut.is_empty())
			{
				res = forward_main(x);
			} else {
				res = forward_main(x) + Shortcut->forward(x);
				if (PostAct != nullptr)
					res = forward_act(PostAct, res, PostActName);
			}
		}
		return res;
	}
};
//TORCH_MODULE(ResidualBlock);
template <typename TMain, typename TShortcut>
using ResidualBlock = torch::nn::ModuleHolder<ResidualBlockImpl<TMain, TShortcut>>;


///
class EfficientViTBlockImpl : public torch::nn::Module {
private:
	ResidualBlock<LiteMLA, IdentityLayer> ContextModule{nullptr};
	std::shared_ptr<torch::nn::AnyModule> LocalModule;
public:
	explicit EfficientViTBlockImpl(
		int in_channels,
		double heads_ratio = 1.0f,
		int dim = 32,
		float expand_ratio = 4.0f,
		const std::vector<int> &scales = { 5 },
		const std::string &norm = "bn2d",
		const std::string &act_func = "hswish",
		const std::string &context_module = "LiteMLA",
		const std::string &local_module = "MBConv"
	) {
		if (context_module == "LiteMLA")
		{
			ContextModule = register_module("context_module", 
				ResidualBlock<LiteMLA, IdentityLayer>(
						LiteMLA(
									in_channels,
									in_channels,
									c10::nullopt,		//heads
									heads_ratio,
									dim,
									false,					//use_bias
									std::make_pair("", norm),
									std::make_pair("", ""),
									"relu",					//kernel_func
									scales
								),
						IdentityLayer()
					)
				);
		} else {
			throw std::invalid_argument("EfficientViTBlockImpl :: context_module is not supported");
		}

		if (local_module == "MBConv")
		{
			LocalModule = std::make_shared<torch::nn::AnyModule>(register_module("local_module",
				ResidualBlock<MBConv, IdentityLayer>(
						MBConv(
									in_channels,
									in_channels,
									3,						//kernel_size
									1,						//stride
									c10::nullopt,	//mid_channels
									expand_ratio,
									std::make_tuple(true, true, false),
									std::make_tuple("", "", norm),
									std::make_tuple(act_func, act_func, "")
								),
						IdentityLayer()
					)
				));
		} else if (local_module == "GLUMBConv") {
			LocalModule = std::make_shared<torch::nn::AnyModule>(register_module("local_module",
				ResidualBlock<GLUMBConv, IdentityLayer>(
					GLUMBConv(
						in_channels,
						in_channels,
						3,							//kernel_size
						1,							//stride
						c10::nullopt,		//mid_channels
						expand_ratio,
						std::make_tuple(true, true, false),
						std::make_tuple("", "", norm),
						std::make_tuple(act_func, act_func, "")
					),
					IdentityLayer()
				)
			));
		} else {
			throw std::invalid_argument("EfficientViTBlockImpl :: local_module is not supported");
		}
	}
	virtual ~EfficientViTBlockImpl() = default;

	torch::Tensor forward(torch::Tensor x)
	{
		x = ContextModule->forward(x);
		x = LocalModule->forward(x);
		return x;
	}
};
TORCH_MODULE(EfficientViTBlock);


struct StackSequentialImpl : public torch::nn::SequentialImpl {
	using SequentialImpl::SequentialImpl;

	torch::Tensor forward(torch::Tensor x)
	{
		return SequentialImpl::forward(x);
	}
};
TORCH_MODULE(StackSequential);


/////Обертка вокруг AnyModule для предоставления доступа к методам
//class ModuleWrapperImpl : public torch::nn::Module {
//private:
//	torch::nn::AnyModule Module;
//public:
//	explicit ModuleWrapperImpl(torch::nn::AnyModule module) : Module(module) {}
//
//	//Переопределение метода forward
//	torch::Tensor forward(torch::Tensor x)
//	{
//		return Module.forward(x);
//	}
//};
//TORCH_MODULE(ModuleWrapper);
