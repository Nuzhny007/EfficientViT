#include "act.h"
#include "norm.h"
#include "opt.h"

#include "safetensors.h"

struct EncoderConfig {
	int in_channels = -1;
	int latent_channels = -1;
	std::vector<int> width_list = { 128, 256, 512, 512, 1024, 1024 };
	std::vector<int> depth_list = { 2, 2, 2, 2, 2, 2 };
	std::variant<std::string, std::vector<std::string>> block_type = "ResBlock";
	//std::variant<std::string, std::vector<std::string>> norm = "trms2d";
	//std::variant<std::string, std::vector<std::string>> act = "silu";
	std::string norm = "trms2d";
	std::string act = "silu";
	std::string downsample_block_type = "ConvPixelUnshuffle";
	bool downsample_match_channel = true;
	/*std::optional<*/std::string/*>*/ downsample_shortcut = "averaging";
	/*std::optional<*/std::string/*>*/ out_norm = ""/* = {}*/;
	/*std::optional<*/std::string/*>*/ out_act = ""/*= {}*/;
	/*std::optional<*/std::string/*>*/ out_shortcut = "averaging";
	bool double_latent = false;
};

struct DecoderConfig {
	int in_channels = -1;
	int latent_channels = -1;
	std::optional<std::string> in_shortcut = "duplicating";
	std::vector<int> width_list = { 128, 256, 512, 512, 1024, 1024 };
	std::vector<int> depth_list = { 2, 2, 2, 2, 2, 2 };
	std::variant<std::string, std::vector<std::string>> block_type = "ResBlock";
	std::variant<std::string, std::vector<std::string>> norm = "trms2d";
	std::variant<std::string, std::vector<std::string>> act = "silu";
	std::string upsample_block_type = "ConvPixelShuffle";
	bool upsample_match_channel = true;
	std::string upsample_shortcut = "duplicating";
	std::string out_norm = "trms2d";
	std::string out_act = "relu";
};

struct DCAEConfig {
	int in_channels;
	int latent_channels;
	EncoderConfig encoder;
	DecoderConfig decoder;
	bool use_quant_conv = false;

	std::string pretrained_path,
		pretrained_source = "dc-ae";

	std::optional<float> scaling_factor = {1.f};

	DCAEConfig(int in_ch = 3,	int latent_ch = 32)
		: in_channels(in_ch), latent_channels(latent_ch)
	{
		encoder.in_channels = in_channels;
		encoder.latent_channels = latent_channels;
		decoder.in_channels = in_channels;
		decoder.latent_channels = latent_channels;
	}
};


///
void add_to_seq_block(
	StackSequential seq,
	const std::string &block_type,
	int in_channels,
	int out_channels,
	const std::optional<std::string> &norm,
	const std::optional<std::string> &act
) {
	if (block_type == "ResBlock")
	{
		if (in_channels != out_channels)
			throw std::invalid_argument("build_block error: For ResBlock, in_channels must be equal to out_channels.");

		auto main_block = /*std::make_shared<*/ResBlock/*>*/(
			in_channels,
			out_channels,
			3/*kernel_size*/,
			1/*stride*/,
			c10::nullopt/*mid_channels*/,
			1.0/*expand_ratio*/,
			std::make_pair(true, false),
			std::make_pair("", norm.value_or("")),
			std::make_pair(act.value_or(""), "")
		);

		auto block = ResidualBlock<ResBlock, IdentityLayer>(main_block, IdentityLayer());
		seq->push_back(block);
	}	else if (block_type == "EViT_GLU") {
		if (in_channels != out_channels)
			throw std::invalid_argument("build_block error: For EViT_GLU, in_channels must be equal to out_channels.");

		auto block = EfficientViTBlock(in_channels, 1.0 /*heads_ratio*/, 32/*dim*/, 4.0f/*expand_ratio*/, std::vector<int>{}/*scales*/, norm.value_or(""), act.value_or(""), std::string("LiteMLA")/*context_module*/, std::string("GLUMBConv")/*local_module*/);
		seq->push_back(block);
	}	else if (block_type == "EViTS5_GLU") {
		if (in_channels != out_channels)
			throw std::invalid_argument("build_block error: For EViTS5_GLU, in_channels must be equal to out_channels.");

		auto block = EfficientViTBlock(in_channels, 1.0 /*heads_ratio*/, 32/*dim*/, 4.0f/*expand_ratio*/, std::vector<int>{5}/*scales*/, norm.value_or(""), act.value_or(""), std::string("LiteMLA")/*context_module*/, std::string("GLUMBConv")/*local_module*/);
		return seq->push_back(block);
	}	else {
		throw std::invalid_argument("build_block error: block_type " + block_type + " is not supported");
	}
}



///
void add_to_seq_stage_main(
	StackSequential seq,
	int width,
	int depth,
	const std::variant<std::string, std::vector<std::string>> &block_type,
	const std::string &norm,
	const std::string &act,
	int input_width
) {

	if (std::holds_alternative<std::string>(block_type))		//Если block_type является строкой
	{
		for (int d = 0; d < depth; d++)
		{
			auto current_block_type = std::get<std::string>(block_type);
			add_to_seq_block(
				seq,
				current_block_type,
				d > 0 ? width : input_width,
				width,
				norm,
				act
			);
		}
	} else if (std::holds_alternative<std::vector<std::string>>(block_type)) {			//Если block_type является вектором строк
		const auto &block_types = std::get<std::vector<std::string>>(block_type);
		if (depth != static_cast<int>(block_types.size()))
			throw std::invalid_argument("add_to_seq_stage_main error: If block_type is a list, its length must be equal to depth.");

		for (int d = 0; d < depth; d++)
		{
			auto current_block_type = block_types[d];
			add_to_seq_block(
				seq,
				current_block_type,
				d > 0 ? width : input_width,
				width,
				norm,
				act
			);
		}
	} else {
		throw std::invalid_argument("add_to_seq_stage_main error: block_type must be a string or a list of strings.");
	}
}


///
void add_to_seq_downsample_block(
	StackSequential seq,
	const std::string &block_type,
	int in_channels,
	int out_channels,
	const std::optional<std::string> &shortcut)
{
	if (block_type == "Conv")
	{
		auto block_module = ConvLayer(
			in_channels,
			out_channels,
			3, // kernel_size
			2, /*stride*/
			1, /*dilation*/
			1, /*groups*/
			true /*use_bias*/,
			0 /*dropuot*/,
			"",	/*norm*/
			""	/*act*/
		);

		if (shortcut.has_value())
		{
			if (shortcut.value() == "averaging")
			{
				auto shortcut_block = PixelUnshuffleChannelAveragingDownSampleLayer(in_channels, out_channels, 2);
				seq->push_back(ResidualBlock<ConvLayer, PixelUnshuffleChannelAveragingDownSampleLayer>(
						block_module,
						shortcut_block
					));
			} else {
				throw std::invalid_argument("build_downsample_block error: shortcut " + *shortcut + " is not supported for downsample");
			}
		} else {
			seq->push_back(block_module);
		}
	} else if (block_type == "ConvPixelUnshuffle") {
		auto block_module = ConvPixelUnshuffleDownSampleLayer(in_channels, out_channels, 3, 2);

		if (shortcut.has_value())
		{
			if (shortcut.value() == "averaging")
			{
				auto shortcut_block = PixelUnshuffleChannelAveragingDownSampleLayer(in_channels, out_channels, 2);
				seq->push_back(
					ResidualBlock<ConvPixelUnshuffleDownSampleLayer, PixelUnshuffleChannelAveragingDownSampleLayer>(
						block_module,
						shortcut_block
					));
			} else {
				throw std::invalid_argument("build_downsample_block error: shortcut " + *shortcut + " is not supported for downsample");
			}
		} else {
			seq->push_back(block_module);
		}
	} else {
		throw std::invalid_argument("build_downsample_block error: block_type " + block_type + " is not supported for downsampling");
	}
}


///
void add_to_seq_upsample_block(
	StackSequential seq,
	const std::string &block_type,
	int in_channels,
	int out_channels,
	const std::optional<std::string> &shortcut
) {

	if (block_type == "ConvPixelShuffle")
	{
		auto cpsus_layer = ConvPixelShuffleUpSampleLayer(in_channels, out_channels, 3, 2);

		if (shortcut.has_value())
		{
			if (shortcut.value() == "duplicating")
			{
				auto shortcut_block = ChannelDuplicatingPixelUnshuffleUpSampleLayer(in_channels, out_channels, 2);
				seq->push_back(
					ResidualBlock<ConvPixelShuffleUpSampleLayer, ChannelDuplicatingPixelUnshuffleUpSampleLayer>(
						cpsus_layer,
						shortcut_block
					)/*.ptr()*/
				);
			} else {
				throw std::invalid_argument("add_to_seq_upsample_block error: shortcut " + *shortcut + " is not supported for upsample");
			}
		} else {
			seq->push_back(cpsus_layer);
		}
	} else if (block_type == "InterpolateConv") {
		auto icus_layer = InterpolateConvUpSampleLayer(in_channels, out_channels, 3, 2);

		if (shortcut.has_value())
		{
			if (shortcut.value() == "duplicating")
			{
				auto shortcut_block = ChannelDuplicatingPixelUnshuffleUpSampleLayer(in_channels, out_channels, 2);
				seq->push_back(
					ResidualBlock<InterpolateConvUpSampleLayer, ChannelDuplicatingPixelUnshuffleUpSampleLayer>(
						icus_layer,
						shortcut_block
					)/*.ptr()*/
				);
			} else {
				throw std::invalid_argument("add_to_seq_upsample_block error: shortcut " + *shortcut + " is not supported for upsample");
			}
		} else {
			seq->push_back(icus_layer);
		}
	} else {
		throw std::invalid_argument("add_to_seq_upsample_block error: block_type " + block_type + " is not supported for upsampling");
	}
}


///
StackSequential build_encoder_project_in_block(
	int in_channels,
	int out_channels,
	int factor,
	const std::string &downsample_block_type
) {
	StackSequential seq;

	if (factor == 1)
	{
		seq->push_back(ConvLayer(
			in_channels,
			out_channels,
			3, /*kernel_size*/
			1, /*stride*/
			1, /*dilation*/
			1, /*groups*/
			true, /*use_bias*/
			0, /*dropout*/
			"",
			""
		));
	} else if (factor == 2) {
		add_to_seq_downsample_block(seq, downsample_block_type, in_channels, out_channels, std::nullopt);
	} else {
		throw std::invalid_argument("build_encoder_project_in_block error: downsample factor " + std::to_string(factor) + " is not supported for encoder project in");
	}

	return seq;
}


///
StackSequential build_encoder_project_out_block(
	const int in_channels,
	const int out_channels,
	const std::string &norm,
	const std::string &act,
	const std::optional<std::string> &shortcut
) {
	StackSequential seq;
	std::shared_ptr<torch::nn::Module> norm_module = build_norm(norm, /*?*/in_channels/*?*/);
	if (norm_module != nullptr)
		add_to_seq_norm(seq, norm_module, norm);		//seq->push_back(norm_module.ptr());

	std::shared_ptr<torch::nn::Module> act_module = build_act(act);
	if (act_module != nullptr)
		add_to_seq_act(seq, act_module, act);		//seq->push_back(act_module);

	seq->push_back(
		ConvLayer(
			in_channels,
			out_channels,
			3, /*kernel_size*/
			1, /*stride*/
			1, /*dilation*/
			1, /*groups*/
			true, /*use_bias*/
			0, /*dropout*/
			"",
			""
		)
	);

	StackSequential result;
	if (shortcut)
	{
		if (*shortcut == "averaging")
		{
			auto shortcut_block = PixelUnshuffleChannelAveragingDownSampleLayer(in_channels, out_channels, 1);
			result->push_back(ResidualBlock<StackSequential, PixelUnshuffleChannelAveragingDownSampleLayer>(
				seq,
				shortcut_block
			));
		} else {
			throw std::invalid_argument("build_downsample_block error: shortcut " + *shortcut + " is not supported for downsample");
		}
	} else {
		result = seq;
	}

	return result;
}


///
StackSequential build_decoder_project_in_block(
	int in_channels,
	int out_channels,
	const std::optional<std::string> &shortcut
){
	StackSequential seq;
	auto block_module = ConvLayer(
			in_channels,
			out_channels,
			3, /*kernel_size*/
			1, /*stride*/
			1, /*dilation*/
			1, /*groups*/
			true, /*use_bias*/
			0, /*dropout*/
			"",
			""
		);

	if (shortcut.has_value())
	{
		if (shortcut.value() == "duplicating")
		{
			auto shortcut_block = ChannelDuplicatingPixelUnshuffleUpSampleLayer(in_channels, out_channels, 1);
			seq->push_back(ResidualBlock<ConvLayer, ChannelDuplicatingPixelUnshuffleUpSampleLayer>(
				block_module,
				shortcut_block
			));
		} else {
			throw std::invalid_argument("build_decoder_project_in_block error: shortcut " + *shortcut + " is not supported for decoder project in");
		}
	} else {
		seq->push_back(block_module);
	}

	return seq;
}


///
StackSequential build_decoder_project_out_block(
	const int in_channels,
	const int out_channels,
	const int factor,
	const std::string &upsample_block_type,
	const std::string &norm,
	const std::string &act
) {
	StackSequential seq;
	std::shared_ptr<torch::nn::Module> norm_module = build_norm(norm, in_channels);
	if (norm_module != nullptr)
		add_to_seq_norm(seq, norm_module, norm);		//seq->push_back(norm_module);

	std::shared_ptr<torch::nn::Module> act_module = build_act(act);
	if (act_module != nullptr)
		add_to_seq_act(seq, act_module, act);		//seq->push_back(act_module);

	if (factor == 1)
	{
		seq->push_back(
			ConvLayer(
				in_channels,
				out_channels,
				3, /*kernel_size*/
				1, /*stride*/
				1, /*dilation*/
				1, /*groups*/
				true, /*use_bias*/
				0, /*dropout*/
				"",
				""
			)
		);
	} else if (factor == 2) {
		add_to_seq_upsample_block(seq, upsample_block_type, in_channels, out_channels, std::nullopt);		//seq->push_back(build_upsample_block(upsample_block_type, in_channels, out_channels, std::nullopt));
	} else {
		throw std::invalid_argument("build_decoder_project_out_block error: upsample factor " + std::to_string(factor) + " is not supported for decoder project out");
	}

	return seq;
}


///
class EncoderImpl : public torch::nn::Module {
private:
	EncoderConfig Config;
	size_t NumStages;
	StackSequential ProjectIn{nullptr},
		ProjectOut{nullptr};
	std::vector<StackSequential> Stages;
public:
	EncoderImpl(const EncoderConfig &cfg) : Config(cfg)
	{
		NumStages = cfg.width_list.size();
		TORCH_CHECK(cfg.depth_list.size() == NumStages);
		TORCH_CHECK(cfg.width_list.size() == NumStages);
		if (std::holds_alternative<std::vector<std::string>>(cfg.block_type))
			TORCH_CHECK(std::get<std::vector<std::string>>(cfg.block_type).size() == NumStages);
		
		ProjectIn = build_encoder_project_in_block(
			cfg.in_channels,
			cfg.depth_list[0] > 0 ? cfg.width_list[0] : cfg.width_list[1],
			cfg.depth_list[0] > 0 ? 1 : 2,
			cfg.downsample_block_type
		);

		for (size_t stage_id = 0; stage_id < NumStages; stage_id++)
		{
			int width = cfg.width_list[stage_id];
			int depth = cfg.depth_list[stage_id];
			std::string block_type;
			if (std::holds_alternative<std::vector<std::string>>(cfg.block_type))
			{
				block_type = std::get<std::vector<std::string>>(cfg.block_type)[stage_id];
			} else {
				block_type = std::get<std::string>(cfg.block_type);
			}

			StackSequential stage;
			add_to_seq_stage_main(stage, width, depth, block_type, cfg.norm, cfg.act, width);

			if ((stage_id < NumStages - 1) && depth > 0)
			{
				int next_width = cfg.downsample_match_channel ? cfg.width_list[stage_id + 1] : width;
				add_to_seq_downsample_block(
					stage,
					cfg.downsample_block_type,
					width,
					next_width,
					cfg.downsample_shortcut
				);
			}
			Stages.push_back(stage);
		}

		ProjectOut = build_encoder_project_out_block(
			cfg.width_list.back(),
			cfg.double_latent ? 2 * cfg.latent_channels : cfg.latent_channels,
			cfg.out_norm,
			cfg.out_act,
			cfg.out_shortcut
		);

		register_module("project_in", ProjectIn);
		for (size_t i = 0; i < Stages.size(); ++i)
			register_module("stages" + std::to_string(i), Stages[i]);
		register_module("project_out", ProjectOut);
	}

	torch::Tensor forward(torch::Tensor x)
	{
		x = ProjectIn->forward(x);
		for (const auto &stage : Stages)
		{
			if (!stage.is_empty() && !stage->is_empty())
			{
				x = stage.ptr()->forward(x);
			}
		}
		x = ProjectOut->forward(x);
		return x;
	}
};
TORCH_MODULE(Encoder);


///
class DecoderImpl : public torch::nn::Module {
private:
	DecoderConfig Config;
	size_t NumStages;
	StackSequential ProjectIn{nullptr},
		ProjectOut{nullptr};
	std::vector<StackSequential> stages_;
public:
	DecoderImpl(const DecoderConfig &cfg) : Config(cfg)
	{
		NumStages = cfg.width_list.size();
		TORCH_CHECK(cfg.depth_list.size() == NumStages);
		TORCH_CHECK(cfg.width_list.size() == NumStages);
		if (std::holds_alternative<std::vector<std::string>>(cfg.block_type))
			TORCH_CHECK(std::get<std::vector<std::string>>(cfg.block_type).size() == NumStages);
		if (std::holds_alternative<std::vector<std::string>>(cfg.norm))
			TORCH_CHECK(std::get<std::vector<std::string>>(cfg.norm).size() == NumStages);
		if (std::holds_alternative<std::vector<std::string>>(cfg.act))
			TORCH_CHECK(std::get<std::vector<std::string>>(cfg.act).size() == NumStages);

		ProjectIn = build_decoder_project_in_block(
			cfg.latent_channels,
			cfg.width_list.back(),
			cfg.in_shortcut
		);

		for (int stage_id = static_cast<int>(NumStages) - 1; stage_id >= 0; stage_id--)
		{
			int width = cfg.width_list[stage_id];
			int depth = cfg.depth_list[stage_id];

			StackSequential stage;
			if ((stage_id < NumStages - 1) && depth > 0)
			{
				int next_width = cfg.upsample_match_channel ? width : cfg.width_list[stage_id + 1];
				add_to_seq_upsample_block(
					stage,
					cfg.upsample_block_type,
					cfg.width_list[stage_id + 1],
					next_width,
					cfg.upsample_shortcut
				);
			}

			std::string block_type;
			if (std::holds_alternative<std::vector<std::string>>(cfg.block_type))
			{
				block_type = std::get<std::vector<std::string>>(cfg.block_type)[stage_id];
			} else {
				block_type = std::get<std::string>(cfg.block_type);
			}

			std::string norm;
			if (std::holds_alternative<std::vector<std::string>>(cfg.norm))
			{
				norm = std::get<std::vector<std::string>>(cfg.norm)[stage_id];
			} else {
				norm = std::get<std::string>(cfg.norm);
			}

			std::string act;
			if (std::holds_alternative<std::vector<std::string>>(cfg.act))
			{
				act = std::get<std::vector<std::string>>(cfg.act)[stage_id];
			} else {
				act = std::get<std::string>(cfg.act);
			}

			int input_width = cfg.upsample_match_channel ? width : cfg.width_list[std::min(size_t(stage_id + 1), NumStages - 1)];
			add_to_seq_stage_main(stage, width, depth, block_type, norm, act, input_width);

			stages_.insert(stages_.begin(), stage);
		}

		ProjectOut = build_decoder_project_out_block(
			cfg.depth_list[0] > 0 ? cfg.width_list[0] : cfg.width_list[1],
			cfg.in_channels,
			cfg.depth_list[0] > 0 ? 1 : 2,
			cfg.upsample_block_type,
			cfg.out_norm,
			cfg.out_act
		);

		register_module("project_in", ProjectIn);		
		//for (size_t i = 0; i < stages_.size(); i++)
		for (int i = static_cast<int>(NumStages) - 1; i >= 0; i--)
		{
			register_module("stages" + std::to_string(i), stages_[i]);
		}
		register_module("project_out", ProjectOut);
	}

	torch::Tensor forward(torch::Tensor x)
	{
		x = ProjectIn->forward(x);
		//reverse
		for (auto it = stages_.rbegin(); it != stages_.rend(); it++)
		{
			if (!(*it).is_empty() && !(*it)->is_empty())
				x = (*it).ptr()->forward(x);
		}
		x = ProjectOut->forward(x);
		return x;
	}

	size_t GetNumStages() { return NumStages; }
};
TORCH_MODULE(Decoder);


///
class DCAEImpl : public torch::nn::Module {
private:
	DCAEConfig cfg_;
public:
	Encoder encoder_{nullptr};
	Decoder decoder_{nullptr};

	DCAEImpl(const DCAEConfig &cfg) : cfg_(cfg)
	{
		encoder_ = Encoder(cfg.encoder);
		decoder_ = Decoder(cfg.decoder);

		register_module("encoder", encoder_);		//???
		register_module("decoder", decoder_);		//???

		for (auto &k : this->named_parameters())
			std::cout << k.key() << std::endl;

		auto ParamsCount = [&]()
		{
			int result = 0;
			for (auto p : this->parameters())
			{
				int ss = 1;
				for (auto s : p.sizes())
					ss *= s;
				result += ss;
			}
			return result;
		};
		std::cout << "Model params count: " << ParamsCount() << std::endl;

		if (cfg.pretrained_path != "")
			load_model(torch::nn::ModuleHolder<DCAEImpl>(*this), cfg.pretrained_source, cfg.pretrained_path, torch::kCPU/*cfg_.Device*/);
	}

	static void load_model(torch::nn::ModuleHolder<DCAEImpl> model, const std::string &pretrained_source, const std::string &pretrained_path, torch::Device device)
	{
		std::cout << "loading model weights..." << std::endl;

		auto modify_param_name = [](const std::string &name)
		{
			std::string modified_name = name;
			size_t pos = 0;
			const std::string stages = "stages";
			//Ищем все вхождения подстроки "stages"
			while ((pos = modified_name.find(stages, pos)) != std::string::npos)
			{
				//Заменяем на "stages."
				modified_name.replace(pos, stages.length(), "stages.");
				pos += stages.length() + 2;
				//Вставляем подстроку ".op_list"
				modified_name.insert(pos, ".op_list");
			}

			const std::string project_in = "project_in";
			pos = 0;
			//Ищем вхождениe подстроки "project_in"
			if ((pos = modified_name.find(project_in, pos)) != std::string::npos)
				if (isdigit(modified_name[pos + project_in.length() + 1]))
				{
					//И удаляем следующие два символа ".0"
					modified_name.replace(pos + project_in.length(), 2, "");
				}

			const std::string project_out = "project_out";
			pos = 0;
			//Ищем вхождениe подстроки "project_out"
			if ((pos = modified_name.find("encoder", pos)) != std::string::npos)
				if ((pos = modified_name.find(project_out, pos)) != std::string::npos)
					if (isdigit(modified_name[pos + project_out.length() + 1]))
					{
						//И удаляем следующие два символа ".0"
						modified_name.replace(pos + project_out.length(), 2, "");
					}

			pos = 0;
			//Ищем вхождениe подстроки "project_out"
			if ((pos = modified_name.find("decoder", pos)) != std::string::npos)
				if ((pos = modified_name.find(project_out, pos)) != std::string::npos)
					if (isdigit(modified_name[pos + project_out.length() + 1]))
					{
						//Вставляем подстроку ".op_list"
						modified_name.insert(pos + project_out.length(), ".op_list");
					}

			const std::string main = "main";
			pos = 0;
			//Ищем вхождениe подстроки "main"
			if ((pos = modified_name.find(main, pos)) != std::string::npos)
				if (isdigit(modified_name[pos + main.length() + 1]))
				{
					//Вставляем подстроку ".op_list"
					modified_name.insert(pos+main.length(), ".op_list");
				}
			return modified_name;
		};


		auto unmodify_param_name = [](const std::string& name)
		{
			std::string unmodified_name = name;
			size_t pos = 0;

			//1.Обработка "stages." и ".op_list"
			const std::string stages = "stages.";
			while ((pos = unmodified_name.find(stages, pos)) != std::string::npos)
			{
				//Заменяем "stages." на "stages"
				unmodified_name.replace(pos, stages.length(), "stages");
			}

			pos = 0;
			const std::string oplist = ".op_list";
			while ((pos = unmodified_name.find(oplist, pos)) != std::string::npos)
			{
				//Удаляем ".op_list"
				unmodified_name.replace(pos, oplist.length(), "");
			}

			//2.Обработка project_in
			const std::string project_in = "project_in";
			pos = 0;
			if ((pos = unmodified_name.find(project_in, pos)) != std::string::npos)
			{
				//Вставляем ".0" после project_in, если следующая часть не начинается с цифры
				if (pos + project_in.length() < unmodified_name.length() &&
					!isdigit(unmodified_name[pos + project_in.length()]))
				{
					unmodified_name.insert(pos + project_in.length(), ".0");
				}
			}

			//3.Обработка project_out в encoder
			const std::string project_out = "project_out";
			pos = 0;
			if ((pos = unmodified_name.find("encoder", pos)) != std::string::npos)
			{
				if ((pos = unmodified_name.find(project_out, pos)) != std::string::npos)
				{
					//Вставляем ".0" после project_out, если следующая часть не начинается с цифры
					if (pos + project_out.length() < unmodified_name.length() &&
						!isdigit(unmodified_name[pos + project_out.length()]))
					{
						unmodified_name.insert(pos + project_out.length(), ".0");
					}
				}
			}

			//4.Обработка project_out в decoder
			pos = 0;
			if ((pos = unmodified_name.find("decoder", pos)) != std::string::npos)
			{
				if ((pos = unmodified_name.find(project_out, pos)) != std::string::npos)
				{
					//Удаляем ".op_list" после project_out, если есть цифра
					const std::string decoder_oplist = ".op_list";
					size_t oplist_pos = pos + project_out.length();
					if (unmodified_name.substr(oplist_pos, decoder_oplist.length()) == decoder_oplist)
					{
						if (oplist_pos + decoder_oplist.length() < unmodified_name.length() &&
							isdigit(unmodified_name[oplist_pos + decoder_oplist.length()]))
						{
							unmodified_name.replace(oplist_pos, decoder_oplist.length(), "");
						}
					}
				}
			}

			//5.Обработка main
			pos = 0;
			const std::string main = "main";
			if ((pos = unmodified_name.find(main, pos)) != std::string::npos)
			{
				// Удаляем ".op_list" после main, если есть цифра
				const std::string main_oplist = ".op_list";
				size_t oplist_pos = pos + main.length();
				if (unmodified_name.substr(oplist_pos, main_oplist.length()) == main_oplist)
				{
					if (oplist_pos + main_oplist.length() < unmodified_name.length() &&
						isdigit(unmodified_name[oplist_pos + main_oplist.length()]))
					{
						unmodified_name.replace(oplist_pos, main_oplist.length(), "");
					}
				}
			}

			return unmodified_name;
		};

		if (pretrained_source == "dc-ae")
		{
			//Загрузка состояния модели из файла .safetensors
			try {
				torch::NoGradGuard no_grad;
				//torch::load(model, pretrained_path);
				std::unordered_map<std::string, torch::Tensor> state_dict = std::move(safetensors::load_safetensors(pretrained_path, device));

				//Загрузка параметров weights & biases
				for (const auto &param : model->named_parameters())
				{
					std::string modified_key = modify_param_name(param.key());
					if (state_dict.find(modified_key) != state_dict.end())		//!!!Можно написать компаратор
					{
						if (param.value().sizes() != state_dict.at(modified_key).sizes())
						{
							std::cout << "Size mismatch for parameter " << modified_key << ". Model expects " << param.value().sizes() <<
								", but loaded " << state_dict.at(modified_key).sizes() << std::endl;
						}
						param.value().detach_().copy_(state_dict.at(modified_key));
					} else {
						throw std::runtime_error("Missing parameter: " + modified_key + " " + param.key());
					}
				}

				//Загрузка буферов running_mean, running_var...
				for (const auto& buffer : model->named_buffers())
				{
					std::string modified_key = modify_param_name(buffer.key());
					if (state_dict.find(modified_key) != state_dict.end())
					{
						if (buffer.value().sizes() != state_dict.at(modified_key).sizes())
						{
							std::cout << "Size mismatch for buffer " << modified_key << ". Model expects " << buffer.value().sizes() <<
								", but loaded " << state_dict.at(modified_key).sizes() << std::endl;
						}
						buffer.value().detach_().copy_(state_dict.at(modified_key));
					} else {
						std::cout << "Warning: Missing buffer " << modified_key << std::endl;
					}
				}
				
				//!!!
				for (const auto &param : state_dict)
				{
					std::string unmodified_key = unmodify_param_name(param.first);
					if (model->named_parameters().find(unmodified_key) == nullptr && model->named_buffers().find(unmodified_key) == nullptr)
					{
						std::cout << param.first << " was not found" << std::endl;
					}
				}

				std::cout << "Loaded " << model->named_parameters().size() << " parameters and " << model->named_buffers().size() << 
					" buffers from " << state_dict.size() << std::endl;
			} catch (std::exception &e) {
				std::cout << e.what() << std::endl;
			}
		} else {
			throw std::runtime_error("DCAEImpl error: NotImplementedError");
		}
	}

	int spatial_compression_ratio() const
	{
		return pow(2, int(decoder_.ptr()->GetNumStages()) - 1);
	}

	torch::Tensor encode(torch::Tensor x)
	{
		x = encoder_->forward(x);
		return x;
	}

	torch::Tensor decode(torch::Tensor x)
	{
		x = decoder_->forward(x);
		return x;
	}

	std::tuple<torch::Tensor, torch::Tensor, std::map<std::string, torch::Tensor>> forward(torch::Tensor x, int global_step)
	{
		x = encoder_->forward(x);
		x = decoder_->forward(x);
		return std::make_tuple(x, torch::tensor(0), std::map<std::string, torch::Tensor>{});
	}
};
TORCH_MODULE(DCAE);



///
DCAEConfig dc_ae_f32c32(const std::string &name, const std::string &pretrained_path)
{
	DCAEConfig cfg;

	if (name == "dc-ae-f32c32-in-1.0" || name == "dc-ae-f32c32-mix-1.0")
	{
		cfg.encoder.latent_channels = 32;
		cfg.encoder.block_type = std::vector<std::string>{ "ResBlock", "ResBlock", "ResBlock", "EViT_GLU", "EViT_GLU", "EViT_GLU" };
		cfg.encoder.width_list = { 128, 256, 512, 512, 1024, 1024 };
		cfg.encoder.depth_list = { 0, 4, 8, 2, 2, 2 };
		cfg.decoder.block_type = std::vector<std::string>{ "ResBlock", "ResBlock", "ResBlock", "EViT_GLU", "EViT_GLU", "EViT_GLU" };
		cfg.decoder.width_list = { 128, 256, 512, 512, 1024, 1024 };
		cfg.decoder.depth_list = { 0, 5, 10, 2, 2, 2 };
		cfg.decoder.norm = std::vector<std::string>{ "bn2d", "bn2d", "bn2d", "trms2d", "trms2d", "trms2d" };
		cfg.decoder.act = std::vector<std::string>{ "relu", "relu", "relu", "silu", "silu", "silu" };
		cfg.scaling_factor = 0.3189;
	} else if (name == "dc-ae-f32c32-sana-1.0") {
		cfg.encoder.latent_channels = 32;
		cfg.encoder.block_type = std::vector<std::string>{ "ResBlock", "ResBlock", "ResBlock", "EViTS5_GLU", "EViTS5_GLU", "EViTS5_GLU" };
		cfg.encoder.width_list = { 128, 256, 512, 512, 1024, 1024 };
		cfg.encoder.depth_list = { 2, 2, 2, 3, 3, 3 };

		cfg.decoder.latent_channels = 32;
		cfg.decoder.block_type = std::vector<std::string>{ "ResBlock", "ResBlock", "ResBlock", "EViTS5_GLU", "EViTS5_GLU", "EViTS5_GLU" };
		cfg.decoder.width_list = { 128, 256, 512, 512, 1024, 1024 };
		cfg.decoder.depth_list = { 3, 3, 3, 3, 3, 3 };
		cfg.decoder.norm = { "trms2d" };
		cfg.decoder.act = { "silu" };
		cfg.scaling_factor = 0.41407;
		cfg.decoder.upsample_block_type = "InterpolateConv";
	} else {
		throw std::runtime_error("NotImplementedError");
	}

	cfg.pretrained_path = pretrained_path;
	return cfg;
}


//def dc_ae_f64c128(name: str, pretrained_path : Optional[str] = None)->DCAEConfig:
//if name in["dc-ae-f64c128-in-1.0", "dc-ae-f64c128-mix-1.0"] :
//	cfg_str = (
//		"latent_channels=128 "
//		"encoder.block_type=[ResBlock,ResBlock,ResBlock,EViT_GLU,EViT_GLU,EViT_GLU,EViT_GLU] "
//		"encoder.width_list=[128,256,512,512,1024,1024,2048] encoder.depth_list=[0,4,8,2,2,2,2] "
//		"decoder.block_type=[ResBlock,ResBlock,ResBlock,EViT_GLU,EViT_GLU,EViT_GLU,EViT_GLU] "
//		"decoder.width_list=[128,256,512,512,1024,1024,2048] decoder.depth_list=[0,5,10,2,2,2,2] "
//		"decoder.norm=[bn2d,bn2d,bn2d,trms2d,trms2d,trms2d,trms2d] decoder.act=[relu,relu,relu,silu,silu,silu,silu]"
//		)
//else:
//raise NotImplementedError
//cfg = OmegaConf.from_dotlist(cfg_str.split(" "))
//cfg : DCAEConfig = OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(DCAEConfig), cfg))
//cfg.pretrained_path = pretrained_path
//return cfg



///
//DCAEConfig dc_ae_f64c128(const std::string& name, const std::optional<std::string>& pretrained_path = std::nullopt) {
//	DCAEConfig cfg;
//
//	if (name == "dc-ae-f64c128-in-1.0" || name == "dc-ae-f64c128-mix-1.0") {
//		cfg.encoder.latent_channels = 128;
//		cfg.encoder.block_type = { "ResBlock", "ResBlock", "ResBlock", "EViT_GLU", "EViT_GLU", "EViT_GLU", "EViT_GLU" };
//		cfg.encoder.width_list = { 128, 256, 512, 512, 1024, 1024, 2048 };
//		cfg.encoder.depth_list = { 0, 4, 8, 2, 2, 2, 2 };
//
//		cfg.decoder.latent_channels = 128;
//		cfg.decoder.block_type = { "ResBlock", "ResBlock", "ResBlock", "EViT_GLU", "EViT_GLU", "EViT_GLU", "EViT_GLU" };
//		cfg.decoder.width_list = { 128, 256, 512, 512, 1024, 1024, 2048 };
//		cfg.decoder.depth_list = { 0, 5, 10, 2, 2, 2, 2 };
//		cfg.decoder.norm = { "bn2d", "bn2d", "bn2d", "trms2d", "trms2d", "trms2d", "trms2d" };
//		cfg.decoder.act = { "relu", "relu", "relu", "silu", "silu", "silu", "silu" };
//	}
//	else {
//		throw std::runtime_error("NotImplementedError");
//	}
//
//	cfg.pretrained_path = pretrained_path;
//	return cfg;
//}



//def dc_ae_f128c512(name: str, pretrained_path : Optional[str] = None)->DCAEConfig:
//if name in["dc-ae-f128c512-in-1.0", "dc-ae-f128c512-mix-1.0"] :
//	cfg_str = (
//		"latent_channels=512 "
//		"encoder.block_type=[ResBlock,ResBlock,ResBlock,EViT_GLU,EViT_GLU,EViT_GLU,EViT_GLU,EViT_GLU] "
//		"encoder.width_list=[128,256,512,512,1024,1024,2048,2048] encoder.depth_list=[0,4,8,2,2,2,2,2] "
//		"decoder.block_type=[ResBlock,ResBlock,ResBlock,EViT_GLU,EViT_GLU,EViT_GLU,EViT_GLU,EViT_GLU] "
//		"decoder.width_list=[128,256,512,512,1024,1024,2048,2048] decoder.depth_list=[0,5,10,2,2,2,2,2] "
//		"decoder.norm=[bn2d,bn2d,bn2d,trms2d,trms2d,trms2d,trms2d,trms2d] decoder.act=[relu,relu,relu,silu,silu,silu,silu,silu]"
//		)
//else:
//raise NotImplementedError
//cfg = OmegaConf.from_dotlist(cfg_str.split(" "))
//cfg : DCAEConfig = OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(DCAEConfig), cfg))
//cfg.pretrained_path = pretrained_path
//return cfg



///
//DCAEConfig dc_ae_f128c512(const std::string& name, const std::optional<std::string>& pretrained_path = std::nullopt) {
//	DCAEConfig cfg;
//
//	if (name == "dc-ae-f128c512-in-1.0" || name == "dc-ae-f128c512-mix-1.0") {
//		cfg.encoder.latent_channels = 512;
//		cfg.encoder.block_type = { "ResBlock", "ResBlock", "ResBlock", "EViT_GLU", "EViT_GLU", "EViT_GLU", "EViT_GLU", "EViT_GLU" };
//		cfg.encoder.width_list = { 128, 256, 512, 512, 1024, 1024, 2048, 2048 };
//		cfg.encoder.depth_list = { 0, 4, 8, 2, 2, 2, 2, 2 };
//
//		cfg.decoder.latent_channels = 512;
//		cfg.decoder.block_type = { "ResBlock", "ResBlock", "ResBlock", "EViT_GLU", "EViT_GLU", "EViT_GLU", "EViT_GLU", "EViT_GLU" };
//		cfg.decoder.width_list = { 128, 256, 512, 512, 1024, 1024, 2048, 2048 };
//		cfg.decoder.depth_list = { 0, 5, 10, 2, 2, 2, 2, 2 };
//		cfg.decoder.norm = { "bn2d", "bn2d", "bn2d", "trms2d", "trms2d", "trms2d", "trms2d", "trms2d" };
//		cfg.decoder.act = { "relu", "relu", "relu", "silu", "silu", "silu", "silu", "silu" };
//	}
//	else {
//		throw std::runtime_error("NotImplementedError");
//	}
//
//	cfg.pretrained_path = pretrained_path;
//	return cfg;
//}
