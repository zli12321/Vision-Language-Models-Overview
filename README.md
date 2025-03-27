# Benchmark and Evaluations, RL Alignment, Applications, and Challenges of Large Vision Language Models
A most Frontend Collection and survey of vision-language model papers, and models GitHub repository

Below we compile *awesome* papers and model and github repositories that 
- **State-of-the-Art VLMs** Collection of newest to oldest VLMs (we'll keep updating new models and benchmarks).
- **Evaluate** VLM benchmarks and corresponding link to the works
- **Post-training/Alignment** Newest related work for VLM alignment including RL, sft.
- **Applications** applications of VLMs in embodied AI, robotics, etc.
- Contribute **surveys**, **perspectives**, and **datasets** on the above topics.


Welcome to contribute and discuss!

---

🤩 Papers marked with a ⭐️ are contributed by the maintainers of this repository. If you find them useful, we would greatly appreciate it if you could give the repository a star or cite our paper.

---

## Table of Contents
* 0. [📄 Paper Link](https://arxiv.org/abs/2501.02189)/[⛑️ Citation](#Citations)
* 1. [📚 SoTA VLMs](#vlms)
* 2. [🗂️ Dataset and Evaluation](#Dataset)
	* 2.1.  [Datasets and Evaluation for VLM](#DatasetforVLM)
	* 2.2.  [Benchmark Datasets, Simulators and Generative Models for Embodied VLM](#DatasetforEmbodiedVLM)
* 3. ### 🔥 [💑 Post-Training/Alignment](#posttraining) 🔥
	* 3.1.  [RL Alignment for VLM](#alignment)
	* 3.2.  [Regular finetuning (SFT)](#sft) 
	* 3.3.  [VLM Alignment Github](#vlm_github)


* 4. [⚒️ Applications](#Toolenhancement)
	* 4.1. 	[Embodied VLM agents](#EmbodiedVLMagents)
	* 4.2.	[Generative Visual Media Applications](#GenerativeVisualMediaApplications)
	* 4.3.	[Robotics and Embodied AI](#RoboticsandEmbodiedAI)
		* 4.3.1.  [Manipulation](#Manipulation)
		* 4.3.2.  [Navigation](#Navigation)
		* 4.3.3.  [Human-robot Interaction](#HumanRobotInteraction)
  		* 4.3.4.  [Autonomous Driving](#AutonomousDriving)
	* 4.4. [Human-Centered AI](#Human-CenteredAI)
		* 4.4.1. [Web Agent](#WebAgent)
		* 4.4.2. [Accessibility](#Accessibility)
		* 4.4.3. [Healthcare](#Healthcare)
		* 4.4.4. [Social Goodness](#SocialGoodness)
* 5. [⛑️ Challenges](#Challenges)
	* 5.1. [Hallucination](#Hallucination)
	* 5.2. [Safety](#Safety)
	* 5.3. [Fairness](#Fairness)
	* 5.4. [Alignment](#Alignment)
  		* 5.4.1. [Multi-modality Alignment](#MultimodalityAlignment)
    		* 5.4.2. [Commonsense and Physics Alignment](#CommonsenseAlignment)
 	* 5.5. [Efficient Training and Fine-Tuning](#EfficientTrainingandFineTuning)
 	* 5.6. [Scarce of High-quality Dataset](#ScarceofHighqualityDataset)


---

##  1. <a name='vlms'></a>📚 SoTA VLMs 
| Model                                                        | Year | Architecture   | Training Data               | Parameters     | Vision Encoder/Tokenizer                       | Pretrained Backbone Model                          |
|--------------------------------------------------------------|------|----------------|-----------------------------|----------------|-----------------------------------------------|---------------------------------------------------|
| [QWen2.5-VL](https://arxiv.org/abs/2502.13923)				   | 2025 | Decdoer-only	|Image caption, VQA, grounding agent, long video | 3B/7B/72B |Redesigned ViT | [Qwen2.5](https://huggingface.co/Qwen)
| [Ola](https://arxiv.org/pdf/2502.04328)					   | 2025 | Decoder-only	|Image/Video/Audio/Text		| 7B			|[OryxViT](https://huggingface.co/THUdyh/Oryx-ViT)| [Qwen-2.5-7B](https://qwenlm.github.io/blog/qwen2.5/), [SigLIP-400M](https://arxiv.org/pdf/2303.15343), [Whisper-V3-Large](https://arxiv.org/pdf/2212.04356), [BEATs-AS2M(cpt2)](https://arxiv.org/pdf/2212.09058)
| [Ocean-OCR](https://arxiv.org/abs/2501.15558)				   | 2025 | Decdoer-only	| Pure Text, Caption, [Interleaved](https://github.com/OpenGVLab/MM-Interleaved), [OCR](https://github.com/X-PLUG/mPLUG-DocOwl/tree/main/DocOwl1.5) | 3B | [NaViT](https://arxiv.org/pdf/2307.06304) | Pretrained from scratch      
| [SmolVLM](https://huggingface.co/blog/smolervlm)             | 2025 | Decoder-only   | [SmolVLM-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-Instruct/blob/main/smolvlm-data.pdf) | 250M & 500M     | SigLIP                                | [SmolLM](https://huggingface.co/blog/smollm)   
| [DeepSeek-Janus-Pro](https://janusai.pro/wp-content/uploads/2025/01/janus_pro_tech_report.pdf)             | 2025 | Decoder-only   | Undisclosed | 7B     | SigLIP                                | [DeepSeek-Janus-Pro](https://huggingface.co/deepseek-ai/Janus-Pro-7B)                                      |
| [Inst-IT](https://arxiv.org/abs/2412.03565) | 2024 | Decoder-only | [Inst-IT Dataset](https://huggingface.co/datasets/Inst-IT/Inst-It-Dataset), [LLaVA-NeXT-Data](https://huggingface.co/datasets/lmms-lab/LLaVA-NeXT-Data) | 7B | CLIP/Vicuna, SigLIP/Qwen2 | [LLaVA-NeXT](https://huggingface.co/liuhaotian/llava-v1.6-vicuna-7b) |
 [DeepSeek-VL2](https://arxiv.org/pdf/2412.10302)             | 2024 | Decoder-only   | [WiT](https://huggingface.co/datasets/google/wit), [WikiHow](https://huggingface.co/datasets/ajibawa-2023/WikiHow) | 4.5B x 74      | SigLIP/SAMB                                  | [DeepSeekMoE](https://arxiv.org/pdf/2412.10302)                                      |
| [xGen-MM (BLIP-3)](https://arxiv.org/pdf/2408.08872) | 2024 | Decoder-only | [MINT-1T](https://arxiv.org/pdf/2406.11271), [OBELICS](https://arxiv.org/pdf/2306.16527), [Caption](https://github.com/salesforce/LAVIS/tree/xgen-mm?tab=readme-ov-file#data-preparation) | 4B | ViT + [Perceiver Resampler](https://arxiv.org/pdf/2204.14198) | [Phi-3-mini](https://arxiv.org/pdf/2404.14219) |
| [TransFusion](https://arxiv.org/pdf/2408.11039)              | 2024 | Encoder-decoder| Undisclosed                 | 7B             | VAE Encoder                                  | Pretrained from scratch on transformer architecture |
| [Baichuan Ocean Mini](https://arxiv.org/pdf/2410.08565)      | 2024 | Decoder-only   | Image/Video/Audio/Text      | 7B             | CLIP ViT-L/14                                | [Baichuan](https://arxiv.org/pdf/2309.10305)                                         |
| [LLaMA 3.2-vision](https://arxiv.org/pdf/2407.21783)         | 2024 | Decoder-only   | Undisclosed                 | 11B-90B        | CLIP                                         | [LLaMA-3.1](https://arxiv.org/pdf/2407.21783)                                        |
| [Pixtral](https://arxiv.org/pdf/2410.07073)                  | 2024 | Decoder-only   | Undisclosed                 | 12B            | CLIP ViT-L/14                                | [Mistral Large 2](https://mistral.ai/)                                  |
| [Qwen2-VL](https://arxiv.org/pdf/2409.12191)                 | 2024 | Decoder-only   | Undisclosed        | 7B-14B         | EVA-CLIP ViT-L                               | [Qwen-2](https://arxiv.org/pdf/2407.10671)                                           |
| [NVLM](https://arxiv.org/pdf/2409.11402)                     | 2024 | Encoder-decoder| [LAION-115M ](https://laion.ai/blog/laion-5b/)      | 8B-24B         | Custom ViT                                   | [Qwen-2-Instruct](https://arxiv.org/pdf/2407.10671)                                  |
| [Emu3](https://arxiv.org/pdf/2409.18869)                     | 2024 | Decoder-only   | [Aquila](https://arxiv.org/pdf/2408.07410)         | 7B             | MoVQGAN                                      | [LLaMA-2](https://arxiv.org/pdf/2307.09288)                                          |
| [Claude 3](https://claude.ai/new)                            | 2024 | Decoder-only   | Undisclosed                 | Undisclosed    | Undisclosed                                  | Undisclosed                                       |
| [InternVL](https://arxiv.org/pdf/2312.14238)                 | 2023 | Encoder-decoder| [LAION-en, LAION- multi](https://laion.ai/blog/laion-5b/)        | 7B/20B         | Eva CLIP ViT-g                               | [QLLaMA](https://arxiv.org/pdf/2304.08177)                                           |
| [InstructBLIP](https://arxiv.org/pdf/2305.06500)             | 2023 | Encoder-decoder| [CoCo](https://cocodataset.org/#home), [VQAv2](https://huggingface.co/datasets/lmms-lab/VQAv2)            | 13B            | ViT                                          | [Flan-T5](https://arxiv.org/pdf/2210.11416), [Vicuna](https://lmsys.org/blog/2023-03-30-vicuna/)                                       |
| [CogVLM](https://arxiv.org/pdf/2311.03079)                   | 2023 | Encoder-decoder| [LAION-2B](https://sisap-challenges.github.io/2024/datasets/) ,[COYO-700M](https://github.com/kakaobrain/coyo-dataset)       | 18B            | CLIP ViT-L/14                                | [Vicuna](https://lmsys.org/blog/2023-03-30-vicuna/)                                                |
| [PaLM-E](https://arxiv.org/pdf/2303.03378)                   | 2023 | Decoder-only   | All robots, [WebLI](https://arxiv.org/pdf/2209.06794)            | 562B           | ViT                                          | [PaLM](https://arxiv.org/pdf/2204.02311)                                             |
| [LLaVA-1.5](https://arxiv.org/pdf/2310.03744)                | 2023 | Decoder-only   | [COCO](https://cocodataset.org/#home)         | 13B            | CLIP ViT-L/14                                | [Vicuna](https://lmsys.org/blog/2023-03-30-vicuna/)                                           |
| [Gemini](https://arxiv.org/pdf/2312.11805)                   | 2023 | Decoder-only   | Undisclosed                 | Undisclosed    | Undisclosed                                  | Undisclosed                                       |
| [GPT-4V](https://arxiv.org/pdf/2309.17421)                   | 2023 | Decoder-only   | Undisclosed                 | Undisclosed    | Undisclosed                                  | Undisclosed                                       |
| [BLIP-2](https://arxiv.org/pdf/2301.12597)                   | 2023 | Encoder-decoder| [COCO](https://cocodataset.org/#home), [Visual Genome](https://huggingface.co/datasets/ranjaykrishna/visual_genome) | 7B-13B         | ViT-g                                        | [Open Pretrained Transformer (OPT)](https://arxiv.org/pdf/2205.01068)                |
| [Flamingo](https://arxiv.org/pdf/2204.14198)                 | 2022 | Decoder-only   | [M3W](https://arxiv.org/pdf/2204.14198), [ALIGN](https://huggingface.co/docs/transformers/en/model_doc/align) | 80B            | Custom                                       | [Chinchilla](https://arxiv.org/pdf/2203.15556)                                        |
| [BLIP](https://arxiv.org/pdf/2201.12086)                     | 2022 | Encoder-decoder| [COCO](https://cocodataset.org/#home), [Visual Genome](https://huggingface.co/datasets/ranjaykrishna/visual_genome/) | 223M-400M      | ViT-B/L/g                                    | Pretrained from scratch                           |
| [CLIP](https://arxiv.org/pdf/2103.00020)                     | 2021 | Encoder-decoder| 400M image-text pairs       | 63M-355M       | ViT/ResNet                                   | Pretrained from scratch                           |
| [VisualBERT](https://arxiv.org/pdf/1908.03557)               | 2019 | Encoder-only   | [COCO](https://cocodataset.org/#home)                        | 110M           | Faster R-CNN                                  | Pretrained from scratch                           |




##  2. <a name='Dataset'></a>🗂️ Benchmarks and Evaluation
### 2.1. <a name='DatasetforVLM'></a> Datasets and Evaluation for VLM
| Benchmark Dataset                                        | Domain                                       | Metric Type        | Source                 | Size (K) | Project |
|----------------------------------------------------------|----------------------------------------------|--------------------|------------------------|----------|---------|
| [Inst-IT-Bench](https://arxiv.org/abs/2412.03565) | Fine-grained Image and Video Understanding | Multiple Choice & LLM Eval | Human/Synthetic | 2K | [Github Repo](https://github.com/inst-it/inst-it) |
| [MovieChat](https://arxiv.org/abs/2307.16449)           | Video understanding             | LLM Eval          | Human                  | 1K       | [Github Repo](https://rese1f.github.io/MovieChat/) |
| [PHYSBENCH](https://arxiv.org/pdf/2501.16411)           | Visual math reasoning                       | Multiple Choice   | Graduate STEM Students | 100      | [Github Repo](https://github.com/USC-GVL/PhysBench) |
| [MMTBench](https://arxiv.org/pdf/2404.16006)           | Visual reasoning, understanding, recognition, and question answering | Multiple Choice | AI Experts | 30.1 | [Github Repo](https://github.com/tylin/coco-caption) |
| [MM-Vet](https://arxiv.org/pdf/2308.02490)             | Optical Character Recognition (OCR) / Visual reasoning | LLM Eval | Human | 0.2 | [Github Repo](https://github.com/yuweihao/MM-Vet) |
| [MM-En/CN](https://arxiv.org/pdf/2307.06281)           | Multilingual multimodal understanding       | Multiple Choice   | Human                  | 3.2      | [Github Repo](https://github.com/open-compass/VLMEvalKit) |
| [GQA](https://arxiv.org/abs/2305.13245)                | Visual reasoning, understanding, recognition, and question answering | Answer Matching | Seed with Synthetic | 22,000 | [Website](https://cs.stanford.edu/people/dorarad/gqa/index.html) |
| [VCR](https://arxiv.org/abs/1811.10830)                | Visual reasoning, understanding, recognition, and question answering | Multiple Choice | MTurks | 290 | [Website](https://visualcommonsense.com/) |
| [VQAv2](https://arxiv.org/pdf/1505.00468)              | Visual reasoning, understanding, recognition, and question answering | Yes/No, Answer Matching | MTurks | 1,100 | [Github Repo](https://github.com/salesforce/LAVIS/blob/main/dataset_card/vqav2.md) |
| [MMMU](https://arxiv.org/pdf/2311.16502)               | Visual reasoning, understanding, recognition, and question answering | Answer Matching, Multiple Choice | College Students | 11.5 | [Website](https://mmmu-benchmark.github.io/) |
| [TextVQA](https://arxiv.org/pdf/1904.08920)           | Visual text understanding                   | Answer Matching   | Expert Human           | 28.6     | [Github Repo](https://github.com/facebookresearch/mmf) |
| [DocVQA](https://arxiv.org/pdf/2007.00398)            | Visual text understanding                   | Answer Matching   | CrowdSource            | 50       | [Website](https://www.docvqa.org/) |
| [MSCOCO-30K](https://arxiv.org/pdf/1405.0312)         | Text-to-Image generation                    | BLEU, Rouge, Similarity | MTurks | 30 | [Website](https://cocodataset.org/#home) |
| [ChartQA](https://arxiv.org/abs/2203.10244)           | Chart graphic understanding                 | Answer Matching   | CrowdSource/Synthetic  | 32.7     | [Github Repo](https://github.com/vis-nlp/ChartQA) |
| [Perception-Test](https://arxiv.org/pdf/2305.13786)   | Video understanding                         | Multiple Choice   | CrowdSource            | 11.6     | [Github Repo](https://github.com/google-deepmind/perception_test) |
| [MMLU](https://arxiv.org/pdf/2009.03300)             | Multimodal general intelligence             | Multiple Choice   | Human                  | 15.9     | [Github Repo](https://github.com/hendrycks/test) |
| [MMStar](https://arxiv.org/pdf/2403.20330)           | Multimodal general intelligence             | Multiple Choice   | Human                  | 1.5      | [Website](https://mmstar-benchmark.github.io/) |
| [VideoMME](https://arxiv.org/pdf/2405.21075)         | Video understanding                         | Multiple Choice   | Experts                | 2.7      | [Website](https://video-mme.github.io/) |
| [EgoSchem](https://arxiv.org/pdf/2308.09126)         | Video understanding                         | Multiple Choice   | Synthetic/Human        | 5        | [Website](https://egoschema.github.io/) |
| [HallusionBench](https://arxiv.org/pdf/2310.14566)   | Hallucination                               | Yes/No            | Human                  | 1.13     | [Github Repo](https://github.com/tianyi-lab/HallusionBench) |
| [POPE](https://arxiv.org/pdf/2305.10355)             | Hallucination                               | Yes/No            | Human                  | 9        | [Github Repo](https://github.com/RUCAIBox/POPE) |
| [CHAIR](https://arxiv.org/pdf/1809.02156)             | Hallucination                               | Yes/No            | Human                  | 124        | [Github Repo](https://github.com/LisaAnne/Hallucination/tree/master) |
| [MHalDetect](https://arxiv.org/abs/2308.06394)| Hallucination | Answer Matching | Human | 4 |                           [Github Repo](https://github.com/LisaAnne/Hallucination/tree/master) |    
| [Hallu-Pi](https://arxiv.org/abs/2408.01355)| Hallucination                 |  Answer Matching  |  Human | 1.260 |      [Github Repo](https://github.com/NJUNLP/Hallu-PI)  
| [HallE-Control](https://arxiv.org/abs/2310.01779) | Hallucination                   |       Yes/No | Human | 108  | [Github Repo](https://github.com/bronyayang/HallE_Control)
| [AutoHallusion](https://arxiv.org/pdf/2406.10900) | Hallucination         |         Answer Matching |  Synthetic |      3.129      | [Github Repo](https://github.com/wuxiyang1996/AutoHallusion) |       
| [BEAF](https://arxiv.org/abs/2407.13442) | Hallucination         |         Yes/No |  Human |      26      | [Github Repo]([https://github.com/wuxiyang1996/AutoHallusion](https://beafbench.github.io/)) |   
| [GAIVE](https://arxiv.org/abs/2306.14565) | Hallucination         |          Answer Matching |  Synthetic |      320      | [Github Repo](https://github.com/FuxiaoLiu/LRV-Instruction) | 
| [HalEval](https://arxiv.org/abs/2402.15721) | Hallucination         | Yes/No |  CrowdSource/Synthetic |     2,000      | [Github Repo](https://github.com/WisdomShell/hal-eval) |    
| [AMBER](https://arxiv.org/abs/2311.07397) | Hallucination         | Answer Matching |  Human |     15.22      | [Github Repo](https://github.com/junyangwang0410/AMBER) |       
| [GenAI-Bench](https://arxiv.org/pdf/2406.13743)      | Text-to-Image generation                    | Human Ratings     | Human                  | 80.0     | [Huggingface](https://huggingface.co/datasets/BaiqiL/GenAI-Bench) |
| [NaturalBench](https://arxiv.org/pdf/2410.14669)     | Multimodal general intelligence             | Yes/No, Multiple Choice | Human | 10.0 | [Huggingface](https://huggingface.co/datasets/BaiqiL/NaturalBench/blob/main/README.md) |
| [R1-Onevision](https://arxiv.org/pdf/2503.10615)     |  Visual reasoning, understanding, recognition | Multiple Choice | Human  | 155 | [Github Repo](https://github.com/Fancy-MLLM/R1-Onevision) |
| [VLM^2-Bench](https://arxiv.org/pdf/2502.12084)      | Visual reasoning, understanding, recognition, and question answering | Answer Matching, Multiple Choice | Human | 3 | [Website](https://vlm2-bench.github.io/) | 
| [VisualWebInstruct](https://arxiv.org/pdf/2503.10582) | Visual reasoning, understanding, recognition, and question answering | LLM Eval |  Web | 900 | [Website](https://tiger-ai-lab.github.io/VisualWebInstruct/) | 


### 2.2. <a name='DatasetforEmbodiedVLM'></a> Benchmark Datasets, Simulators, and Generative Models for Embodied VLM 
| Benchmark                                                                                                                                     |             Domain              |                Type                |                                                     		Project					                                                     |
|-----------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------:|:----------------------------------:|:----------------------------------------------------------------------------------------------------------------------:|
| [Habitat](https://arxiv.org/pdf/1904.01201), [Habitat 2.0](https://arxiv.org/pdf/2106.14405), [Habitat 3.0](https://arxiv.org/pdf/2310.13724) |      Robotics (Navigation)      |        Simulator + Dataset         |                                           [Website](https://aihabitat.org/)                                            |
| [Gibson](https://arxiv.org/pdf/1808.10654)                                                                                                    |      Robotics (Navigation)      |        Simulator + Dataset         |           [Website](http://gibsonenv.stanford.edu/), [Github Repo](https://github.com/StanfordVL/GibsonEnv)            |
| [iGibson1.0](https://arxiv.org/pdf/2012.02924), [iGibson2.0](https://arxiv.org/pdf/2108.03272)                                                |      Robotics (Navigation)      |        Simulator + Dataset         |            [Website](https://svl.stanford.edu/igibson/), [Document](https://stanfordvl.github.io/iGibson/)             |
| [Isaac Gym](https://arxiv.org/pdf/2108.10470)                                                                                                 |      Robotics (Navigation)      |             Simulator              |      [Website](https://developer.nvidia.com/isaac-gym), [Github Repo](https://github.com/isaac-sim/IsaacGymEnvs)       |
| [Isaac Lab](https://arxiv.org/pdf/2301.04195)                                                                                                 |      Robotics (Navigation)      |             Simulator              | [Website](https://isaac-sim.github.io/IsaacLab/main/index.html), [Github Repo](https://github.com/isaac-sim/IsaacLab)  |
| [AI2THOR](https://arxiv.org/abs/1712.05474) |  Robotics (Navigation)      |             Simulator | [Website](https://ai2thor.allenai.org/), [Github Repo](https://github.com/allenai/ai2thor)  |
| [ProcTHOR](https://arxiv.org/abs/2206.06994) |  Robotics (Navigation)      |              Simulator + Dataset | [Website](https://procthor.allenai.org/), [Github Repo](https://github.com/allenai/procthor)  |
| [VirtualHome](https://arxiv.org/abs/1806.07011) |  Robotics (Navigation)      |              Simulator | [Website](http://virtual-home.org/), [Github Repo](https://github.com/xavierpuigf/virtualhome)  |
| [ThreeDWorld](https://arxiv.org/abs/2007.04954) | Robotics (Navigation)      |              Simulator | [Website](https://www.threedworld.org/), [Github Repo](https://github.com/threedworld-mit/tdw)  |
| [VIMA-Bench](https://arxiv.org/pdf/2210.03094)                                                                                                |     Robotics (Manipulation)     |             Simulator              |                [Website](https://vimalabs.github.io/), [Github Repo](https://github.com/vimalabs/VIMA)                 |
| [VLMbench](https://arxiv.org/pdf/2206.08522)                                                                                                  |     Robotics (Manipulation)     |             Simulator              |                                 [Github Repo](https://github.com/eric-ai-lab/VLMbench)                                 |
| [CALVIN](https://arxiv.org/pdf/2112.03227)                                                                                                    |     Robotics (Manipulation)     |             Simulator              |              [Website](http://calvin.cs.uni-freiburg.de/), [Github Repo](https://github.com/mees/calvin)               |
| [GemBench](https://arxiv.org/pdf/2410.01345)                                                                                                  |     Robotics (Manipulation)     |             Simulator              | [Website](https://www.di.ens.fr/willow/research/gembench/), [Github Repo](https://github.com/vlc-robot/robot-3dlotus/) | 
| [WebArena](https://arxiv.org/pdf/2307.13854)                                                                                                  |            Web Agent            |             Simulator              |                [Website](https://webarena.dev/), [Github Repo](https://github.com/web-arena-x/webarena)                |
| [UniSim](https://openreview.net/pdf?id=sFyTZEqmUY)                                                                                            |     Robotics (Manipulation)     |   Generative Model, World Model    |                                [Website](https://universal-simulator.github.io/unisim/)                                |
| [GAIA-1](https://arxiv.org/pdf/2309.17080)                                                                                                    | Robotics (Automonous Driving)   |   Generative Model, World Model    |                                [Website](https://wayve.ai/thinking/introducing-gaia1/)                                 |                                                                                                   
| [LWM](https://arxiv.org/pdf/2402.08268)                                                                                                       |           Embodied AI           |   Generative Model, World Model    |        [Website](https://largeworldmodel.github.io/lwm/), [Github Repo](https://github.com/LargeWorldModel/LWM)        |
| [Genesis](https://github.com/Genesis-Embodied-AI/Genesis)                                                                                     |           Embodied AI           |   Generative Model, World Model    |                             [Github Repo](https://github.com/Genesis-Embodied-AI/Genesis)                              |
| [EMMOE](https://arxiv.org/pdf/2503.08604) | Embodied AI | Generative Model, World Model | [Paper](https://arxiv.org/pdf/2503.08604)  |
| [RoboGen](https://arxiv.org/pdf/2311.01455) | Embodied AI | Generative Model, World Model | [Website](https://robogen-ai.github.io/)  |


##  3. <a name='posttraining'></a>⚒️ Post-Training
### 3.1.  <a name='alignment'></a>RL Alignment for VLM
| Title | Year | Paper | RL | Code |
|----------------|------|--------|---------|------|
| OmniAlign-V: Towards Enhanced Alignment of MLLMs with Human Preference | 2025 | [Paper](https://arxiv.org/abs/2502.18411) | DPO | [Code](https://github.com/PhoenixZ810/OmniAlign-V) |
| Multimodal Open R1/R1-Multimodal-Journey | 2025 | - | GRPO | [Code](https://github.com/EvolvingLMMs-Lab/open-r1-multimodal) |
| R1-VL: Learning to Reason with Multimodal Large Language Models via Step-wise Group Relative Policy Optimization | 2025 | [Paper](https://arxiv.org/abs/2503.12937) | GRPO | [Code](https://github.com/jingyi0000/R1-VL) |
| Agent-R1: Training Powerful LLM Agents with End-to-End Reinforcement Learning | 2025 | - | PPO/REINFORCE++/GRPO | [Code](https://github.com/0russwest0/Agent-R1) |
| R1-VL: Learning to Reason with Multimodal Large Language Models via Step-wise Group Relative Policy Optimization | 2025 | [Paper](https://arxiv.org/abs/2503.12937) | GRPO | [Code](https://github.com/jingyi0000/R1-VL) |
| MM-Eureka: Exploring Visual Aha Moment with Rule-based Large-scale Reinforcement Learning | 2025 | [Paper](https://arxiv.org/abs/2503.07365) | [REINFORCE Leave-One-Out (RLOO)](https://openreview.net/pdf?id=r1lgTGL5DE) | [Code](https://github.com/ModalMinds/MM-EUREKA) |
| MM-RLHF: The Next Step Forward in Multimodal LLM Alignment | 2025 | [Paper](https://arxiv.org/abs/2502.10391) | DPO | [Code](https://github.com/Kwai-YuanQi/MM-RLHF) |
| LMM-R1: Empowering 3B LMMs with Strong Reasoning Abilities Through Two-Stage Rule-Based RL | 2025 | [Paper](https://arxiv.org/pdf/2503.07536) | PPO | [Code](https://github.com/TideDra/lmm-r1) |
| Vision-R1: Incentivizing Reasoning Capability in Multimodal Large Language Models | 2025 | [Paper](https://arxiv.org/pdf/2503.06749) | GRPO | [Code](https://github.com/Osilly/Vision-R1) |
| Unified Reward Model for Multimodal Understanding and Generation | 2025 | [Paper](https://arxiv.org/abs/2503.05236) | DPO | [Code](https://github.com/CodeGoat24/UnifiedReward) |
| Can We Generate Images with CoT? Let's Verify and Reinforce Image Generation Step by Step | 2025 | [Paper](https://arxiv.org/pdf/2501.13926) | DPO | [Code](https://github.com/ZiyuGuo99/Image-Generation-CoT) |
| All Roads Lead to Likelihood: The Value of Reinforcement Learning in Fine-Tuning | 2025 | [Paper](https://arxiv.org/pdf/2503.01067) | Online RL | - |


### 3.2. <a name='sft'></a>Finetuning for VLM
| Title | Year | Paper | Website | Code |
|----------------|------|--------|---------|------|
| Inst-IT: Boosting Multimodal Instance Understanding via Explicit Visual Prompt Instruction Tuning | 2024 | [Paper](https://arxiv.org/abs/2412.03565) | [Website](https://inst-it.github.io/) | [Code](https://github.com/inst-it/inst-it) |
| LLaVolta: Efficient Multi-modal Models via Stage-wise Visual Context Compression | 2024 | [Paper](https://arxiv.org/pdf/2406.20092) | [Website](https://beckschen.github.io/llavolta.html) | [Code](https://github.com/Beckschen/LLaVolta) |
| ViTamin: Designing Scalable Vision Models in the Vision-Language Era | 2024 | [Paper](https://arxiv.org/pdf/2404.02132) | [Website](https://beckschen.github.io/vitamin.html) | [Code](https://github.com/Beckschen/ViTamin) |
| Espresso: High Compression For Rich Extraction From Videos for Your Vision-Language Model | 2024 | [Paper](https://arxiv.org/pdf/2412.04729) | - | - |
| Should VLMs be Pre-trained with Image Data? | 2025 | [Paper](https://arxiv.org/pdf/2503.07603) | - | - |

### 3.3. <a name='vlm_github'></a>VLM Alignment github
| Project | Repository Link |
|----------------|----------------|
| LLaMAFactory | [🔗 GitHub](https://github.com/hiyouga/LLaMA-Factory) |
| MM-Eureka-Zero | [🔗 GitHub](https://github.com/ModalMinds/MM-EUREKA/tree/main) |
| MM-RLHF | [🔗 GitHub](https://github.com/Kwai-YuanQi/MM-RLHF) |
| LMM-R1 | [🔗 GitHub](https://github.com/TideDra/lmm-r1) |


## 4. <a name='Toolenhancement'></a> ⚒️ Applications

### 4.1 Embodied VLM Agents

| Title | Year | Paper Link |
|----------------|------|------------|
| Aligning Cyber Space with Physical World: A Comprehensive Survey on Embodied AI | 2024 | [Paper](https://arxiv.org/pdf/2407.06886v1) |
| ScreenAI: A Vision-Language Model for UI and Infographics Understanding | 2024 | [Paper](https://arxiv.org/pdf/2402.04615) |
| ChartLlama: A Multimodal LLM for Chart Understanding and Generation | 2023 | [Paper](https://arxiv.org/pdf/2311.16483) |
| SciDoc2Diagrammer-MAF: Towards Generation of Scientific Diagrams from Documents guided by Multi-Aspect Feedback Refinement | 2024 | [📄 Paper](https://arxiv.org/pdf/2409.19242) |
| Training a Vision Language Model as Smartphone Assistant | 2024 | [Paper](https://arxiv.org/pdf/2404.08755) |
| ScreenAgent: A Vision-Language Model-Driven Computer Control Agent | 2024 | [Paper](https://arxiv.org/pdf/2402.07945) |
| Embodied Vision-Language Programmer from Environmental Feedback | 2024 | [Paper](https://arxiv.org/pdf/2310.08588) |
| VLMs Play StarCraft II: A Benchmark and Multimodal Decision Method | 2025 | [📄 Paper](https://arxiv.org/abs/2503.05383) | - | [💾 Code](https://github.com/camel-ai/VLM-Play-StarCraft2) |
| MP-GUI: Modality Perception with MLLMs for GUI Understanding | 2025 | [📄 Paper](https://arxiv.org/pdf/2503.14021) | - | [💾 Code](https://github.com/BigTaige/MP-GUI) | 


### 4.2. <a name='GenerativeVisualMediaApplications'></a>Generative Visual Media Applications
| Title | Year | Paper | Website | Code |
|----------------|------|--------|---------|------|
| GPT4Motion: Scripting Physical Motions in Text-to-Video Generation via Blender-Oriented GPT Planning | 2023 | [📄 Paper](https://arxiv.org/pdf/2311.12631) | [🌍 Website](https://gpt4motion.github.io/) | [💾 Code](https://github.com/jiaxilv/GPT4Motion) |
| Spurious Correlation in Multimodal LLMs | 2025 | [📄 Paper](https://arxiv.org/abs/2503.08884) | - | - |
| WeGen: A Unified Model for Interactive Multimodal Generation as We Chat | 2025 |  [📄 Paper](https://arxiv.org/pdf/2503.01115) | - | [💾 Code](https://github.com/hzphzp/WeGen) |
| VideoMind: A Chain-of-LoRA Agent for Long Video Reasoning | 2025 | [📄 Paper](https://arxiv.org/pdf/2503.13444) | [🌍 Website](https://videomind.github.io/) | [💾 Code](https://github.com/yeliudev/VideoMind) |

### 4.3. <a name='RoboticsandEmbodiedAI'></a>Robotics and Embodied AI
| Title | Year | Paper | Website | Code |
|----------------|------|--------|---------|------|
| AHA: A Vision-Language-Model for Detecting and Reasoning Over Failures in Robotic Manipulation | 2024 | [📄 Paper](https://arxiv.org/pdf/2410.00371) | [🌍 Website](https://aha-vlm.github.io/) | - |
| SpatialVLM: Endowing Vision-Language Models with Spatial Reasoning Capabilities | 2024 | [📄 Paper](https://arxiv.org/pdf/2401.12168) | [🌍 Website](https://spatial-vlm.github.io/) | - |
| Vision-language model-driven scene understanding and robotic object manipulation | 2024 | [📄 Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10711845&casa_token=to4vCckCewMAAAAA:2ykeIrubUOxwJ1rhwwakorQFAwUUBQhL_Ct7dnYBceWU5qYXiCoJp_yQkmJbmtiEVuX2jcpvB92n&tag=1) | - | - |
| Guiding Long-Horizon Task and Motion Planning with Vision Language Models | 2024 | [📄 Paper](https://arxiv.org/pdf/2410.02193) | [🌍 Website](https://zt-yang.github.io/vlm-tamp-robot/) | - |
| AutoTAMP: Autoregressive Task and Motion Planning with LLMs as Translators and Checkers | 2023 | [📄 Paper](https://arxiv.org/pdf/2306.06531) | [🌍 Website](https://yongchao98.github.io/MIT-REALM-AutoTAMP/) | - |
| VLM See, Robot Do: Human Demo Video to Robot Action Plan via Vision Language Model | 2024 | [📄 Paper](https://arxiv.org/pdf/2410.08792) | - | - |
| Scalable Multi-Robot Collaboration with Large Language Models: Centralized or Decentralized Systems? | 2023 | [📄 Paper](https://arxiv.org/pdf/2309.15943) | [🌍 Website](https://yongchao98.github.io/MIT-REALM-Multi-Robot/) | - |
| DART-LLM: Dependency-Aware Multi-Robot Task Decomposition and Execution using Large Language Models | 2024 | [📄 Paper](https://arxiv.org/pdf/2411.09022) | [🌍 Website](https://wyd0817.github.io/project-dart-llm/) | - |
| MotionGPT: Human Motion as a Foreign Language | 2023 | [📄 Paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/3fbf0c1ea0716c03dea93bb6be78dd6f-Paper-Conference.pdf) | - | [💾 Code](https://github.com/OpenMotionLab/MotionGPT) |
| Learning Reward for Robot Skills Using Large Language Models via Self-Alignment | 2024 | [📄 Paper](https://arxiv.org/pdf/2405.07162) | - | - |
| Language to Rewards for Robotic Skill Synthesis | 2023 | [📄 Paper](https://language-to-reward.github.io/assets/l2r.pdf) | [🌍 Website](https://language-to-reward.github.io/) | - |
| Eureka: Human-Level Reward Design via Coding Large Language Models | 2023 | [📄 Paper](https://arxiv.org/pdf/2310.12931) | [🌍 Website](https://eureka-research.github.io/) | - |
| Integrated Task and Motion Planning | 2020 | [📄 Paper](https://arxiv.org/pdf/2010.01083) | - | - |
| Jailbreaking LLM-Controlled Robots | 2024 | [📄 Paper](https://arxiv.org/pdf/2410.13691) | [🌍 Website](https://robopair.org/) | - |
| Robots Enact Malignant Stereotypes | 2022 | [📄 Paper](https://arxiv.org/pdf/2207.11569) | [🌍 Website](https://sites.google.com/view/robots-enact-stereotypes) | - |
| LLM-Driven Robots Risk Enacting Discrimination, Violence, and Unlawful Actions | 2024 | [📄 Paper](https://arxiv.org/pdf/2406.08824) | - | - |
| Highlighting the Safety Concerns of Deploying LLMs/VLMs in Robotics | 2024 | [📄 Paper](https://arxiv.org/pdf/2402.10340) | [🌍 Website](https://wuxiyang1996.github.io/adversary-vlm-robotics/) | - |
| EmbodiedBench: Comprehensive Benchmarking Multi-modal Large Language Models for Vision-Driven Embodied Agents | 2025 | [📄 Paper](https://arxiv.org/pdf/2502.09560) | [🌍 Website](https://embodiedbench.github.io/) | [💾 Code & Dataset](https://github.com/EmbodiedBench/EmbodiedBench) |
| Gemini Robotics: Bringing AI into the Physical World | 2025 | [📄 Technical Report](https://storage.googleapis.com/deepmind-media/gemini-robotics/gemini_robotics_report.pdf) | [🌍 Website](https://deepmind.google/technologies/gemini-robotics/) | - |
| GR-2: A Generative Video-Language-Action Model with Web-Scale Knowledge for Robot Manipulation | 2024 | [📄 Paper](https://arxiv.org/pdf/2410.06158) | [🌍 Website](https://gr2-manipulation.github.io/) | - |
| Magma: A Foundation Model for Multimodal AI Agents | 2025 | [📄 Paper](https://arxiv.org/pdf/2502.13130) | [🌍 Website](https://microsoft.github.io/Magma/) | [💾 Code](https://github.com/microsoft/Magma) |
| DayDreamer: World Models for Physical Robot Learning | 2022 | [📄 Paper](https://arxiv.org/pdf/2206.14176)| [🌍 Website](https://danijar.com/project/daydreamer/) | [💾 Code](https://github.com/danijar/daydreamer) |
| Hi Robot: Open-Ended Instruction Following with Hierarchical Vision-Language-Action Models | 2025 | [📄 Paper](https://arxiv.org/pdf/2206.14176)| - | - |
| RL-VLM-F: Reinforcement Learning from Vision Language Foundation Model Feedback | 2024 | [📄 Paper](https://arxiv.org/pdf/2402.03681)| [🌍 Website](https://rlvlmf2024.github.io/) | [💾 Code](https://github.com/yufeiwang63/RL-VLM-F) |
| KALIE: Fine-Tuning Vision-Language Models for Open-World Manipulation without Robot Data | 2024 | [📄 Paper](https://arxiv.org/pdf/2409.14066)| [🌍 Website](https://kalie-vlm.github.io/) | [💾 Code](https://github.com/gractang/kalie) |
| Unified Video Action Model | 2025 | [📄 Paper](https://arxiv.org/pdf/2503.00200)| [🌍 Website](https://unified-video-action-model.github.io/) | [💾 Code](https://github.com/ShuangLI59/unified_video_action) |
| HybridVLA: Collaborative Diffusion and Autoregression in a Unified Vision-Language-Action Model | 2025 | [📄 Paper](https://arxiv.org/abs/2503.10631)| [🌍 Website](https://hybrid-vla.github.io/) | [💾 Code](https://github.com/PKU-HMI-Lab/Hybrid-VLA) |

#### 4.3.1. <a name='Manipulation'></a>Manipulation
| Title | Year | Paper | Website | Code |
|----------------|------|--------|---------|------|
| VIMA: General Robot Manipulation with Multimodal Prompts | 2022 | [📄 Paper](https://arxiv.org/pdf/2210.03094) | [🌍 Website](https://vimalabs.github.io/) |
| Instruct2Act: Mapping Multi-Modality Instructions to Robotic Actions with Large Language Model | 2023 | [📄 Paper](https://arxiv.org/pdf/2305.11176) | - | - |
| Creative Robot Tool Use with Large Language Models | 2023 | [📄 Paper](https://arxiv.org/pdf/2310.13065) | [🌍 Website](https://creative-robotool.github.io/) | - |
| RoboVQA: Multimodal Long-Horizon Reasoning for Robotics | 2024 | [📄 Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10610216) | - | - |
| RT-1: Robotics Transformer for Real-World Control at Scale | 2022 | [📄 Paper](https://robotics-transformer1.github.io/assets/rt1.pdf) | [🌍 Website](https://robotics-transformer1.github.io/) | - |
| RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control | 2023 | [📄 Paper](https://arxiv.org/pdf/2307.15818) | [🌍 Website](https://robotics-transformer2.github.io/) | - |
| Open X-Embodiment: Robotic Learning Datasets and RT-X Models | 2023 | [📄 Paper](https://arxiv.org/pdf/2310.08864) | [🌍 Website](https://robotics-transformer-x.github.io/) | - |
| ExploRLLM: Guiding Exploration in Reinforcement Learning with Large Language Models | 2024 | [📄 Paper](https://arxiv.org/pdf/2403.09583) | [🌍 Website](https://explorllm.github.io/) | - |
| AnyTouch: Learning Unified Static-Dynamic Representation across Multiple Visuo-tactile Sensors | 2025 | [📄 Paper](https://arxiv.org/pdf/2502.12191) | [🌍 Website](https://gewu-lab.github.io/AnyTouch/) | [💾 Code](https://github.com/GeWu-Lab/AnyTouch) |
| Masked World Models for Visual Control | 2022 | [📄 Paper](https://arxiv.org/pdf/2206.14244)| [🌍 Website](https://sites.google.com/view/mwm-rl) | [💾 Code](https://github.com/younggyoseo/MWM) |
| Multi-View Masked World Models for Visual Robotic Manipulation | 2023 | [📄 Paper](https://arxiv.org/pdf/2302.02408)| [🌍 Website](https://sites.google.com/view/mv-mwm) | [💾 Code](https://github.com/younggyoseo/MV-MWM) |


#### 4.3.2. <a name='Navigation'></a>Navigation
| Title | Year | Paper | Website | Code |
|----------------|------|--------|---------|------|
| ZSON: Zero-Shot Object-Goal Navigation using Multimodal Goal Embeddings | 2022 | [📄 Paper](https://arxiv.org/pdf/2206.12403) | - | - |
| LOC-ZSON: Language-driven Object-Centric Zero-Shot Object Retrieval and Navigation | 2024 | [📄 Paper](https://arxiv.org/pdf/2405.05363) | - | - |
| LM-Nav: Robotic Navigation with Large Pre-Trained Models of Language, Vision, and Action | 2022 | [📄 Paper](https://arxiv.org/pdf/2207.04429) | [🌍 Website](https://sites.google.com/view/lmnav) | - |
| NaVILA: Legged Robot Vision-Language-Action Model for Navigation | 2022 | [📄 Paper](https://arxiv.org/pdf/2412.04453) | [🌍 Website](https://navila-bot.github.io/) | - |
| VLFM: Vision-Language Frontier Maps for Zero-Shot Semantic Navigation | 2024 | [📄 Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10610712&casa_token=qvFCSt20n0MAAAAA:MSC4P7bdlfQuMRFrmIl706B-G8ejcxH9ZKROKETL1IUZIW7m_W4hKW-kWrxw-F8nykoysw3WYHnd) | - | - |
| Navigation with Large Language Models: Semantic Guesswork as a Heuristic for Planning | 2023 | [📄 Paper](https://arxiv.org/pdf/2310.10103) | [🌍 Website](https://sites.google.com/view/lfg-nav/) | - |
| Vi-LAD: Vision-Language Attention Distillation for Socially-Aware Robot Navigation in Dynamic Environments | 2025 | [📄 Paper](https://arxiv.org/pdf/2503.09820) | - | - |
| Navigation World Models | 2024 | [📄 Paper](https://arxiv.org/pdf/2412.03572) | [🌍 Website](https://www.amirbar.net/nwm/) | - |


#### 4.3.3. <a name='HumanRobotInteraction'></a>Human-robot Interaction
| Title | Year | Paper | Website | Code |
|----------------|------|--------|---------|------|
| MUTEX: Learning Unified Policies from Multimodal Task Specifications | 2023 | [📄 Paper](https://arxiv.org/pdf/2309.14320) | [🌍 Website](https://ut-austin-rpl.github.io/MUTEX/) | - |
| LaMI: Large Language Models for Multi-Modal Human-Robot Interaction | 2024 | [📄 Paper](https://arxiv.org/pdf/2401.15174) | [🌍 Website](https://hri-eu.github.io/Lami/) | - |
| VLM-Social-Nav: Socially Aware Robot Navigation through Scoring using Vision-Language Models | 2024 | [📄 Paper](https://arxiv.org/pdf/2404.00210) | - | - |

#### 4.3.4. <a name='AutonomousDriving'></a>Autonomous Driving
| Title | Year | Paper | Website | Code |
|----------------|------|--------|---------|------|
| DriveVLM: The Convergence of Autonomous Driving and Large Vision-Language Models | 2024 | [📄 Paper](https://arxiv.org/abs/2402.12289) | [🌍 Website](https://tsinghua-mars-lab.github.io/DriveVLM/) | - |
| GPT-Driver: Learning to Drive with GPT | 2023 | [📄 Paper](https://arxiv.org/abs/2310.01415) | - | - |
| LanguageMPC: Large Language Models as Decision Makers for Autonomous Driving | 2023 | [📄 Paper](https://arxiv.org/abs/2310.03026) | [🌍 Website](https://sites.google.com/view/llm-mpc) | - |
| Driving with LLMs: Fusing Object-Level Vector Modality for Explainable Autonomous Driving | 2023 | [📄 Paper](https://arxiv.org/abs/2310.01957) | - | - |
| Referring Multi-Object Tracking | 2023 | [📄 Paper](https://arxiv.org/pdf/2303.03366) | - | [💾 Code](https://github.com/wudongming97/RMOT) |
| VLPD: Context-Aware Pedestrian Detection via Vision-Language Semantic Self-Supervision | 2023 | [📄 Paper](https://arxiv.org/pdf/2304.03135) | - | [💾 Code](https://github.com/lmy98129/VLPD) |
| MotionLM: Multi-Agent Motion Forecasting as Language Modeling | 2023 | [📄 Paper](https://arxiv.org/pdf/2309.16534) | - | - |
| DiLu: A Knowledge-Driven Approach to Autonomous Driving with Large Language Models | 2023 | [📄 Paper](https://arxiv.org/abs/2309.16292) | [🌍 Website](https://pjlab-adg.github.io/DiLu/) | - |
| VLP: Vision Language Planning for Autonomous Driving | 2024 | [📄 Paper](https://arxiv.org/pdf/2401.05577) | - | - |
| DriveGPT4: Interpretable End-to-end Autonomous Driving via Large Language Model | 2023 | [📄 Paper](https://arxiv.org/abs/2310.01412) | - | - |


### 4.4. <a name='Human-CenteredAI'></a>Human-Centered AI
| Title | Year | Paper | Website | Code |
|----------------|------|--------|---------|------|
| DLF: Disentangled-Language-Focused Multimodal Sentiment Analysis | 2024 | [📄 Paper](https://arxiv.org/pdf/2412.12225) | - | [💾 Code](https://github.com/pwang322/DLF) |
| LIT: Large Language Model Driven Intention Tracking for Proactive Human-Robot Collaboration – A Robot Sous-Chef Application | 2024 | [📄 Paper](https://arxiv.org/abs/2406.13787) | - | - |
| Pretrained Language Models as Visual Planners for Human Assistance | 2023 | [📄 Paper](https://arxiv.org/pdf/2304.09179) | - | - |
| Promoting AI Equity in Science: Generalized Domain Prompt Learning for Accessible VLM Research | 2024 | [📄 Paper](https://arxiv.org/pdf/2405.08668) | - | - |
| Image and Data Mining in Reticular Chemistry Using GPT-4V | 2023 | [📄 Paper](https://arxiv.org/pdf/2312.05468) | - | - |

#### 4.4.1. <a name='WebAgent'></a>Web Agent
| Title | Year | Paper | Website | Code |
|----------------|------|--------|---------|------|
| A Real-World WebAgent with Planning, Long Context Understanding, and Program Synthesis | 2023 | [📄 Paper](https://arxiv.org/pdf/2307.12856) | - | - |
| CogAgent: A Visual Language Model for GUI Agents | 2023 | [📄 Paper](https://arxiv.org/pdf/2312.08914) | - | [💾 Code](https://github.com/THUDM/CogAgent) |
| WebVoyager: Building an End-to-End Web Agent with Large Multimodal Models | 2024 | [📄 Paper](https://arxiv.org/pdf/2401.13919) | - | [💾 Code](https://github.com/MinorJerry/WebVoyager) |
| ShowUI: One Vision-Language-Action Model for GUI Visual Agent | 2024 | [📄 Paper](https://arxiv.org/pdf/2411.17465) | - | [💾 Code](https://github.com/showlab/ShowUI) |
| ScreenAgent: A Vision Language Model-driven Computer Control Agent | 2024 | [📄 Paper](https://arxiv.org/pdf/2402.07945) | - | [💾 Code](https://github.com/niuzaisheng/ScreenAgent) |
| Web Agents with World Models: Learning and Leveraging Environment Dynamics in Web Navigation | 2024 | [📄 Paper](https://arxiv.org/pdf/2410.13232) | - | [💾 Code](https://huggingface.co/papers/2410.13232) |


#### 4.4.2. <a name='Accessibility'></a>Accessibility
| Title | Year | Paper | Website | Code |
|----------------|------|--------|---------|------|
| X-World: Accessibility, Vision, and Autonomy Meet | 2021 | [📄 Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhang_X-World_Accessibility_Vision_and_Autonomy_Meet_ICCV_2021_paper.pdf) | - | - |
| Context-Aware Image Descriptions for Web Accessibility | 2024 | [📄 Paper](https://arxiv.org/pdf/2409.03054) | - | - |
| Improving VR Accessibility Through Automatic 360 Scene Description Using Multimodal Large Language Models | 2024 | [📄 Paper](https://dl.acm.org/doi/10.1145/3691573.3691619) | - | -


#### 4.4.3. <a name='Healthcare'></a>Healthcare
| Title | Year | Paper | Website | Code |
|----------------|------|--------|---------|------|
| VisionUnite: A Vision-Language Foundation Model for Ophthalmology Enhanced with Clinical Knowledge | 2024 | [📄 Paper](https://arxiv.org/pdf/2408.02865) | - | [💾 Code](https://github.com/HUANGLIZI/VisionUnite) |
| Multimodal Healthcare AI: Identifying and Designing Clinically Relevant Vision-Language Applications for Radiology | 2024 | [📄 Paper](https://arxiv.org/pdf/2402.14252) | - | - |
| M-FLAG: Medical Vision-Language Pre-training with Frozen Language Models and Latent Space Geometry Optimization | 2023 | [📄 Paper](https://arxiv.org/pdf/2307.08347) | - | - |
| MedCLIP: Contrastive Learning from Unpaired Medical Images and Text | 2022 | [📄 Paper](https://arxiv.org/pdf/2210.10163) | - | [💾 Code](https://github.com/RyanWangZf/MedCLIP) |
| Med-Flamingo: A Multimodal Medical Few-Shot Learner | 2023 | [📄 Paper](https://arxiv.org/pdf/2307.15189) | - | [💾 Code](https://github.com/snap-stanford/med-flamingo) |


#### 4.4.4. <a name='SocialGoodness'></a>Social Goodness
| Title | Year | Paper | Website | Code |
|----------------|------|--------|---------|------|
| Analyzing K-12 AI Education: A Large Language Model Study of Classroom Instruction on Learning Theories, Pedagogy, Tools, and AI Literacy | 2024 | [📄 Paper](https://www.sciencedirect.com/science/article/pii/S2666920X24000985) | - | - |
| Students Rather Than Experts: A New AI for Education Pipeline to Model More Human-Like and Personalized Early Adolescence | 2024 | [📄 Paper](https://arxiv.org/pdf/2410.15701) | - | - |
| Harnessing Large Vision and Language Models in Agriculture: A Review | 2024 | [📄 Paper](https://arxiv.org/pdf/2407.19679) | - | - |
| A Vision-Language Model for Predicting Potential Distribution Land of Soybean Double Cropping | 2024 | [📄 Paper](https://www.frontiersin.org/journals/environmental-science/articles/10.3389/fenvs.2024.1515752/abstract) | - | - |
| Vision-Language Model is NOT All You Need: Augmentation Strategies for Molecule Language Models | 2024 | [📄 Paper](https://arxiv.org/pdf/2407.09043) | - | [💾 Code](https://github.com/Namkyeong/AMOLE) |
| DrawEduMath: Evaluating Vision Language Models with Expert-Annotated Students’ Hand-Drawn Math Images | 2024 | [📄 Paper](https://openreview.net/pdf?id=0vQYvcinij) | - | - |
| MultiMath: Bridging Visual and Mathematical Reasoning for Large Language Models | 2024 | [📄 Paper](https://arxiv.org/pdf/2409.00147) | - | [💾 Code](https://github.com/pengshuai-rin/MultiMath) |
| Vision-Language Models Meet Meteorology: Developing Models for Extreme Weather Events Detection with Heatmaps | 2024 | [📄 Paper](https://arxiv.org/pdf/2406.09838) | - | [💾 Code](https://github.com/AlexJJJChen/Climate-Zoo) |
| He is Very Intelligent, She is Very Beautiful? On Mitigating Social Biases in Language Modeling and Generation | 2021 | [📄 Paper](https://aclanthology.org/2021.findings-acl.397.pdf) | - | - |
| UrbanVLP: Multi-Granularity Vision-Language Pretraining for Urban Region Profiling | 2024 | [📄 Paper](https://arxiv.org/pdf/2403.168318) | - | - |


## 5. <a name='Challenges'></a>Challenges
### 5.1 <a name='Hallucination'></a>Hallucination
| Title | Year | Paper | Website | Code |
|----------------|------|--------|---------|------|
| Object Hallucination in Image Captioning | 2018 | [📄 Paper](https://arxiv.org/pdf/1809.02156) | - | - |
| Evaluating Object Hallucination in Large Vision-Language Models | 2023 | [📄 Paper](https://arxiv.org/pdf/2305.10355) | - | [💾 Code](https://github.com/RUCAIBox/POPE) |
| Detecting and Preventing Hallucinations in Large Vision Language Models | 2023 | [📄 Paper](https://arxiv.org/pdf/2308.06394) | - | - |
| HallE-Control: Controlling Object Hallucination in Large Multimodal Models | 2023 | [📄 Paper](https://arxiv.org/pdf/2310.01779) | - | [💾 Code](https://github.com/bronyayang/HallE_Control) |
| Hallu-PI: Evaluating Hallucination in Multi-modal Large Language Models within Perturbed Inputs | 2024 | [📄 Paper](https://arxiv.org/pdf/2408.01355) | - | [💾 Code](https://github.com/NJUNLP/Hallu-PI) |
| BEAF: Observing BEfore-AFter Changes to Evaluate Hallucination in Vision-Language Models | 2024 | [📄 Paper](https://arxiv.org/pdf/2407.13442) | [🌍 Website](https://beafbench.github.io/) | - |
| HallusionBench: An Advanced Diagnostic Suite for Entangled Language Hallucination and Visual Illusion in Large Vision-Language Models | 2023 | [📄 Paper](https://arxiv.org/pdf/2310.14566) | - | [💾 Code](https://github.com/tianyi-lab/HallusionBench) |
| AUTOHALLUSION: Automatic Generation of Hallucination Benchmarks for Vision-Language Models | 2024 | [📄 Paper](https://arxiv.org/pdf/2406.10900) | [🌍 Website](https://wuxiyang1996.github.io/autohallusion_page/) | - |
| Mitigating Hallucination in Large Multi-Modal Models via Robust Instruction Tuning | 2023 | [📄 Paper](https://arxiv.org/pdf/2306.14565) | - | [💾 Code](https://github.com/FuxiaoLiu/LRV-Instruction) |
| Hal-Eval: A Universal and Fine-grained Hallucination Evaluation Framework for Large Vision Language Models | 2024 | [📄 Paper](https://arxiv.org/pdf/2402.15721) | - | [💾 Code](https://github.com/WisdomShell/hal-eval) |
| AMBER: An LLM-free Multi-dimensional Benchmark for MLLMs Hallucination Evaluation | 2023 | [📄 Paper](https://arxiv.org/pdf/2311.07397) | - | [💾 Code](https://github.com/junyangwang0410/AMBER) |


### 5.2 <a name='Safety'></a>Safety
| Title | Year | Paper | Website | Code |
|----------------|------|--------|---------|------|
| JailbreakZoo: Survey, Landscapes, and Horizons in Jailbreaking Large Language and Vision-Language Models | 2024 | [📄 Paper](https://arxiv.org/pdf/2407.01599) | [🌍 Website](https://chonghan-chen.com/llm-jailbreak-zoo-survey/) | - |
| Safe-VLN: Collision Avoidance for Vision-and-Language Navigation of Autonomous Robots Operating in Continuous Environments | 2023 | [📄 Paper](https://arxiv.org/pdf/2311.02817) | - | - |
| SafeBench: A Safety Evaluation Framework for Multimodal Large Language Models | 2024 | [📄 Paper](https://arxiv.org/pdf/2410.18927) | - | - |
| JailBreakV: A Benchmark for Assessing the Robustness of MultiModal Large Language Models against Jailbreak Attacks | 2024 | [📄 Paper](https://arxiv.org/pdf/2404.03027) | - | - |
| SHIELD: An Evaluation Benchmark for Face Spoofing and Forgery Detection with Multimodal Large Language Models | 2024 | [📄 Paper](https://arxiv.org/pdf/2402.04178) | - | [💾 Code](https://github.com/laiyingxin2/SHIELD) |
| Images are Achilles' Heel of Alignment: Exploiting Visual Vulnerabilities for Jailbreaking Multimodal Large Language Models | 2024 | [📄 Paper](https://arxiv.org/pdf/2403.09792) | - | - |
| Jailbreaking Attack against Multimodal Large Language Model | 2024 | [📄 Paper](https://arxiv.org/pdf/2402.02309) | - | - |
| Embodied Red Teaming for Auditing Robotic Foundation Models | 2025 | [📄 Paper](https://arxiv.org/pdf/2411.18676) | [🌍 Website](https://s-karnik.github.io/embodied-red-team-project-page/) | [💾 Code](https://github.com/Improbable-AI/embodied-red-teaming) |
| Safety Guardrails for LLM-Enabled Robots | 2025 | [📄 Paper](https://arxiv.org/pdf/2503.07885) | - | - |


### 5.3 <a name='Fairness'></a>Fairness
| Title | Year | Paper | Website | Code |
|----------------|------|--------|---------|------|
| Hallucination of Multimodal Large Language Models: A Survey | 2024 | [📄 Paper](https://arxiv.org/pdf/2404.18930) | - | - |
| Bias and Fairness in Large Language Models: A Survey | 2023 | [📄 Paper](https://arxiv.org/pdf/2309.00770) | - | - |
| Fairness and Bias in Multimodal AI: A Survey | 2024 | [📄 Paper](https://arxiv.org/pdf/2406.19097) | - | - |
| Multi-Modal Bias: Introducing a Framework for Stereotypical Bias Assessment beyond Gender and Race in Vision–Language Models | 2023 | [📄 Paper](http://gerard.demelo.org/papers/multimodal-bias.pdf) | - | - |
| FMBench: Benchmarking Fairness in Multimodal Large Language Models on Medical Tasks | 2024 | [📄 Paper](https://arxiv.org/pdf/2410.01089) | - | - |
| FairCLIP: Harnessing Fairness in Vision-Language Learning | 2024 | [📄 Paper](https://arxiv.org/pdf/2403.19949) | - | - |
| FairMedFM: Fairness Benchmarking for Medical Imaging Foundation Models | 2024 | [📄 Paper](https://arxiv.org/pdf/2407.00983) | - | - |
| Benchmarking Vision Language Models for Cultural Understanding | 2024 | [📄 Paper](https://arxiv.org/pdf/2407.10920) | - | - |

#### 5.4 <a name='Alignment'></a>Alignment
#### 5.4.1 <a name='MultimodalityAlignment'></a>Multi-modality Alignment
| Title | Year | Paper | Website | Code |
|----------------|------|--------|---------|------|
| Mitigating Hallucinations in Large Vision-Language Models with Instruction Contrastive Decoding | 2024 | [📄 Paper](https://arxiv.org/pdf/2403.18715) | - | - |
| Enhancing Visual-Language Modality Alignment in Large Vision Language Models via Self-Improvement | 2024 | [📄 Paper](https://arxiv.org/pdf/2405.15973) | - | - |
| Assessing and Learning Alignment of Unimodal Vision and Language Models | 2024 | [📄 Paper](https://arxiv.org/pdf/2412.04616) | [🌍 Website](https://lezhang7.github.io/sail.github.io/) | - |
| Extending Multi-modal Contrastive Representations | 2023 | [📄 Paper](https://arxiv.org/pdf/2310.08884) | - | [💾 Code](https://github.com/MCR-PEFT/Ex-MCR) |
| OneLLM: One Framework to Align All Modalities with Language | 2023 | [📄 Paper](https://arxiv.org/pdf/2312.03700) | - | [💾 Code](https://github.com/csuhan/OneLLM) |
| What You See is What You Read? Improving Text-Image Alignment Evaluation | 2023 | [📄 Paper](https://arxiv.org/pdf/2305.10400) | [🌍 Website](https://wysiwyr-itm.github.io/) | [💾 Code](https://github.com/yonatanbitton/wysiwyr) |
| Critic-V: VLM Critics Help Catch VLM Errors in Multimodal Reasoning | 2024 | [📄 Paper](https://arxiv.org/pdf/2411.18203) | [🌍 Website](https://huggingface.co/papers/2411.18203) | [💾 Code](https://github.com/kyrieLei/Critic-V) |

#### 5.4.2 <a name='CommonsenseAlignment'></a>Commonsense and Physics Alignment
| Title | Year | Paper | Website | Code |
|----------------|------|--------|---------|------|
| VBench: Comprehensive BenchmarkSuite for Video Generative Models | 2023 | [📄 Paper](https://arxiv.org/pdf/2311.17982) | [🌍 Website](https://vchitect.github.io/VBench-project/) | [💾 Code](https://github.com/Vchitect/VBench) |
| VBench++: Comprehensive and Versatile Benchmark Suite for Video Generative Models | 2024 | [📄 Paper](https://arxiv.org/pdf/2411.13503) | [🌍 Website](https://vchitect.github.io/VBench-project/) | [💾 Code](https://github.com/Vchitect/VBench) |
| PhysBench: Benchmarking and Enhancing VLMs for Physical World Understanding | 2025 | [📄 Paper](https://arxiv.org/pdf/2501.16411) | [🌍 Website](https://physbench.github.io/) | [💾 Code](https://github.com/USC-GVL/PhysBench) | 
| VideoPhy: Evaluating Physical Commonsense for Video Generation | 2024 | [📄 Paper](https://arxiv.org/pdf/2406.03520) | [🌍 Website](https://videophy.github.io/) | [💾 Code](https://github.com/Hritikbansal/videophy) | 
| WorldSimBench: Towards Video Generation Models as World Simulators | 2024 | [📄 Paper](https://arxiv.org/pdf/2410.18072) | [🌍 Website](https://iranqin.github.io/WorldSimBench.github.io/) | - |
| WorldModelBench: Judging Video Generation Models As World Models | 2025 | [📄 Paper](https://arxiv.org/pdf/2502.20694) | [🌍 Website](https://worldmodelbench-team.github.io/) | [💾 Code](https://github.com/WorldModelBench-Team/WorldModelBench/tree/main?tab=readme-ov-file) |
| VideoScore: Building Automatic Metrics to Simulate Fine-grained Human Feedback for Video Generation | 2024 | [📄 Paper](https://arxiv.org/pdf/2406.15252) | [🌍 Website](https://tiger-ai-lab.github.io/VideoScore/) | [💾 Code](https://github.com/TIGER-AI-Lab/VideoScore) |
| WISE: A World Knowledge-Informed Semantic Evaluation for Text-to-Image Generation | 2025 | [📄 Paper](https://arxiv.org/pdf/2503.07265) | - | [💾 Code](https://github.com/PKU-YuanGroup/WISE) |
| Content-Rich AIGC Video Quality Assessment via Intricate Text Alignment and Motion-Aware Consistency | 2025 | [📄 Paper](https://arxiv.org/pdf/2502.04076) | - | [💾 Code](https://github.com/littlespray/CRAVE) |
| Your Large Vision-Language Model Only Needs A Few Attention Heads For Visual Grounding | 2025 | [📄 Paper](https://arxiv.org/pdf/2503.06287) | - | - |
| SpatialVLM: Endowing Vision-Language Models with Spatial Reasoning Capabilities | 2024 | [📄 Paper](https://arxiv.org/pdf/2401.12168) | [🌍 Website](https://spatial-vlm.github.io/) | [💾 Code](https://github.com/remyxai/VQASynth) |
| Do generative video models understand physical principles? | 2025 | [📄 Paper](https://arxiv.org/pdf/2501.09038) | [🌍 Website](https://physics-iq.github.io/) | [💾 Code](https://github.com/google-deepmind/physics-IQ-benchmark) |
| PhysGen: Rigid-Body Physics-Grounded Image-to-Video Generation | 2024 | [📄 Paper](https://arxiv.org/pdf/2409.18964) | [🌍 Website](https://stevenlsw.github.io/physgen/) | [💾 Code](https://github.com/stevenlsw/physgen) |
| How Far is Video Generation from World Model: A Physical Law Perspective | 2024 | [📄 Paper](https://arxiv.org/pdf/2411.02385) | [🌍 Website](https://phyworld.github.io/) | [💾 Code](https://github.com/phyworld/phyworld) |
| Imagine while Reasoning in Space: Multimodal Visualization-of-Thought | 2025 | [📄 Paper](https://arxiv.org/abs/2501.07542) | - | - |

### 5.5 <a name=' EfficientTrainingandFineTuning'></a> Efficient Training and Fine-Tuning
| Title | Year | Paper | Website | Code |
|----------------|------|--------|---------|------|
| VILA: On Pre-training for Visual Language Models | 2023 | [📄 Paper](https://arxiv.org/pdf/2312.07533) | - | - |
| SimVLM: Simple Visual Language Model Pretraining with Weak Supervision | 2021 | [📄 Paper](https://arxiv.org/pdf/2108.10904) | - | - |
| LoRA: Low-Rank Adaptation of Large Language Models | 2021 | [📄 Paper](https://arxiv.org/pdf/2106.09685) | - | [💾 Code](https://github.com/microsoft/LoRA) |
| QLoRA: Efficient Finetuning of Quantized LLMs | 2023 | [📄 Paper](https://arxiv.org/pdf/2305.14314) | - | - |
| Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback | 2022 | [📄 Paper](https://arxiv.org/pdf/2204.05862) | - | [💾 Code](https://github.com/anthropics/hh-rlhf) |
| RLAIF vs. RLHF: Scaling Reinforcement Learning from Human Feedback with AI Feedback | 2023 | [📄 Paper](https://arxiv.org/pdf/2309.00267) | - | - |


### 5.6 <a name='ScarceofHighqualityDataset'></a>Scarce of High-quality Dataset
| Title | Year | Paper | Website | Code |
|----------------|------|--------|---------|------|
| Inst-IT: Boosting Multimodal Instance Understanding via Explicit Visual Prompt Instruction Tuning | 2024 | [📄 Paper](https://arxiv.org/abs/2412.03565) | [Website](https://inst-it.github.io/) | [💾 Code](https://github.com/inst-it/inst-it) |
| SLIP: Self-supervision meets Language-Image Pre-training | 2021 | [📄 Paper](https://arxiv.org/pdf/2112.12750) | - | [💾 Code](https://github.com/facebookresearch/SLIP) |
| Synthetic Vision: Training Vision-Language Models to Understand Physics | 2024 | [📄 Paper](https://arxiv.org/pdf/2412.08619) | - | - |
| Synth2: Boosting Visual-Language Models with Synthetic Captions and Image Embeddings | 2024 | [📄 Paper](https://arxiv.org/pdf/2403.07750) | - | - |
| KALIE: Fine-Tuning Vision-Language Models for Open-World Manipulation without Robot Data | 2024 | [📄 Paper](https://arxiv.org/pdf/2409.14066) | - | - |
| Web Agents with World Models: Learning and Leveraging Environment Dynamics in Web Navigation | 2024 | [📄 Paper](https://arxiv.org/pdf/2410.13232) | - | - |



## 6. <a name='Citations'></a>Citation

```
@misc{li2025surveystateartlarge,
      title={A Survey of State of the Art Large Vision Language Models: Alignment, Benchmark, Evaluations and Challenges}, 
      author={Zongxia Li and Xiyang Wu and Hongyang Du and Huy Nghiem and Guangyao Shi},
      year={2025},
      eprint={2501.02189},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2501.02189}, 
}
```
