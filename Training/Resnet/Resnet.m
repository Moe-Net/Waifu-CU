block0[nChannels_, stride_] := NetGraph[
	<|
		"res_branch1" -> ConvolutionLayer[nChannels, 1, "Stride" -> stride],
		"bn_branch1" -> BatchNormalizationLayer["Epsilon" -> 0.0001],
		"res_branch2a" -> ConvolutionLayer[nChannels / 4, 1, "Stride" -> stride],
		"bn_branch2a" -> BatchNormalizationLayer["Epsilon" -> 0.0001],
		"res_branch2a_relu" -> ElementwiseLayer[Ramp],
		"res_branch2b" -> ConvolutionLayer[nChannels / 4, 3, "PaddingSize" -> 1],
		"bn_branch2b" -> BatchNormalizationLayer["Epsilon" -> 0.0001],
		"res_branch2b_relu" -> ElementwiseLayer[Ramp],
		"res_branch2c" -> ConvolutionLayer[nChannels, 1],
		"bn_branch2c" -> BatchNormalizationLayer["Epsilon" -> 0.0001],
		"res" -> TotalLayer[],
		"res_relu" -> ElementwiseLayer[Ramp]
	|>,
	{
		NetPort["Input"] -> "res_branch1" -> "bn_branch1" -> "res" -> "res_relu",
		NetPort["Input"]
			-> "res_branch2a" -> "bn_branch2a" -> "res_branch2a_relu"
			-> "res_branch2b" -> "bn_branch2b" -> "res_branch2b_relu"
			-> "res_branch2c" -> "bn_branch2c" -> "res"
	}
]
blockN[nChannels_] := NetGraph[
	<|
		"res_branch2a" -> ConvolutionLayer[nChannels / 4, 1],
		"bn_branch2a" -> BatchNormalizationLayer["Epsilon" -> 0.0001],
		"res_branch2a_relu" -> ElementwiseLayer[Ramp],
		"res_branch2b" -> ConvolutionLayer[nChannels / 4, 3, "PaddingSize" -> 1],
		"bn_branch2b" -> BatchNormalizationLayer["Epsilon" -> 0.0001],
		"res_branch2b_relu" -> ElementwiseLayer[Ramp],
		"res_branch2c" -> ConvolutionLayer[nChannels, 1],
		"bn_branch2c" -> BatchNormalizationLayer["Epsilon" -> 0.0001],
		"res" -> TotalLayer[],
		"res_relu" -> ElementwiseLayer[Ramp]
	|>,
	{
		NetPort["Input"] -> "res" -> "res_relu",
		NetPort["Input"]
			-> "res_branch2a" -> "bn_branch2a" -> "res_branch2a_relu"
			-> "res_branch2b" -> "bn_branch2b" -> "res_branch2b_relu"
			-> "res_branch2c" -> "bn_branch2c" -> "res"
	}
]
blockChain[names_, nChannels_, stride_] := Association@Prepend[
	Thread[Rest[names] -> Table[blockN[nChannels], {i, Length[names] - 1}]],
	First[names] -> block0[nChannels, stride]
]
extractor = NetChain[
	Join[<|
		"conv1" -> ConvolutionLayer[64, 7, "Stride" -> 2, "PaddingSize" -> 3],
		"bn_conv1" -> BatchNormalizationLayer["Epsilon" -> 0.0001],
		"conv1_relu" -> ElementwiseLayer[Ramp],
		"pool1_pad" -> PaddingLayer[{{0, 0}, {0, 1}, {0, 1}}, "Padding" -> "Fixed"],
		"pool1" -> PoolingLayer[3, "Stride" -> 2]
	|>,
		blockChain[{"2a", "2b", "2c"}, 256, 1],
		blockChain[{"3a", "3b", "3c", "3d"}, 512, 2],
		blockChain[{"4a", "4b", "4c", "4d", "4e", "4f"}, 1024, 2],
		blockChain[{"5a", "5b", "5c"}, 2048, 2],
		<|"pool5" -> PoolingLayer[7, "Function" -> Mean]|>
	]
]
predictor = NetChain[{
	FlattenLayer[],
	LinearLayer[500],
	Ramp,
	LinearLayer[174],
	SoftmaxLayer[]
}]
res50 = NetChain[
	{extractor, predictor},
	"Input" -> NetEncoder[{"Image", {224, 224}}],
	"Output" -> NetDecoder[{"Class", Append[cap /@ tags, _]}]
]