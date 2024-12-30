# Awesome-VLM-Papers-And-Models
A most Frontend Collection and survey of vision-language model papers, and models GitHub repository

Below we compile *awesome* papers and model and github repositories that 
- **State-of-the-Art VLMs** Collection of VLMs from 2022-2024.
- **applications** applications of VLMs in embodied AI, robotics, etc.
- **evaluate** VLM benchmarks and corresponding link to the works
- contribute **surveys**, **perspectives**, and **datasets** on the above topics.


Welcome to contribute and discuss!

---

🤩 Papers marked with a ⭐️ are contributed by the maintainers of this repository. If you find them useful, we would greatly appreciate it if you could give the repository a star or cite our paper.

---

## Table of Contents

* 1. [📚 SoTA VLMs](#vlms)
* 2. [🗂️ Dataset and Evaluation](#Dataset)
* 4. [⚒️ Applications](#Toolenhancement)
	* 4.1. 	[Embodied VLM agents](#EmbodiedVLMagents)
	* 4.2.	[Generative Visual Media Applications](GenerativeVisualMediaApplications)
	* 4.3.	[Robotics and Embodied AI](RoboticsandEmbodiedAI)
	* 4.4.	[Human-Centered AI](#Human-CenteredAI)
* 5. [⛑️ Alignment](#Alignment)
	* 5.1. [🌈 Pluralistic Alignment](#PluralisticAlignment)
* 6. [🚀 Simulation](#Simulation)
* 7. [👁️‍🗨️ Perspective](#Perspective)


---

##  1. <a name='vlms'></a>📚 SoTA VLMs 
| Model                                                        | Year | Architecture   | Training Data               | Parameters     | Vision Encoder/Tokenizer                       | Pretrained Backbone Model                          |
|--------------------------------------------------------------|------|----------------|-----------------------------|----------------|-----------------------------------------------|---------------------------------------------------|
| [VisualBERT](https://arxiv.org/pdf/1908.03557)                                                   | 2019 | Encoder-only   | COCO                        | 110M           | Faster R-CNN                                  | Pretrained from scratch                           |
| [CLIP](https://arxiv.org/pdf/2103.00020)                                                         | 2021 | Encoder-decoder| 400M image-text pairs       | 63M-355M       | ViT/ResNet                                   | Pretrained from scratch                           |
| [BLIP](https://arxiv.org/pdf/2201.12086)                                                         | 2022 | Encoder-decoder| 129M image-text pairs       | 223M-400M      | ViT-B/L/g                                    | Pretrained from scratch                           |
| [Flamingo](https://arxiv.org/pdf/2204.14198)                                                     | 2022 | Decoder-only   | 185M images                 | 80B            | Custom                                       | Chinchilla                                        |
| [BLIP-2](https://arxiv.org/pdf/2301.12597)                                                       | 2023 | Encoder-decoder| 129M image-text pairs       | 7B-13B         | ViT-g                                        | Open Pretrained Transformer (OPT)                |
| [GPT-4V](https://arxiv.org/pdf/2309.17421)                                                       | 2023 | Decoder-only   | Undisclosed                 | Undisclosed    | Undisclosed                                  | Undisclosed                                       |
| [Gemini](https://arxiv.org/pdf/2312.11805)                                                       | 2023 | Decoder-only   | Undisclosed                 | Undisclosed    | Undisclosed                                  | Undisclosed                                       |
| [LLaVA-1.5](https://arxiv.org/pdf/2310.03744)                                                  | 2023 | Decoder-only   | 558K image-text pairs       | 13B            | CLIP ViT-L/14                                | Vicuna                                           |
| [PaLM-E](https://arxiv.org/pdf/2303.03378)                                                       | 2023 | Decoder-only   | Multiple sources            | 562B           | ViT                                          | PaLM                                             |
| [CogVLM](https://arxiv.org/pdf/2311.03079)                                                       | 2023 | Encoder-decoder| 100M image-text pairs       | 18B            | CLIP ViT-L/14                                | Vicuna                                           |
| [InstructBLIP](https://arxiv.org/pdf/2305.06500)                                                 | 2023 | Encoder-decoder| Multiple sources            | 13B            | ViT                                          | Flan-T5, Vicuna                                  |
| [InternVL](https://arxiv.org/pdf/2312.14238)                                                     | 2023 | Encoder-decoder| 800M image-text pairs       | 7B/20B         | Eva CLIP ViT-g                               | QLLaMA                                           |
| [Claude 3](https://claude.ai/new)                                                     | 2024 | Decoder-only   | Undisclosed                 | Undisclosed    | Undisclosed                                  | Undisclosed                                       |
| [Emu3](https://arxiv.org/pdf/2409.18869)                                                         | 2024 | Decoder-only   | 1B image-text pairs         | 7B             | MoVQGAN                                      | LLaMA-2                                     |
| [NVLM](https://arxiv.org/pdf/2409.11402)                                                         | 2024 | Encoder-decoder| 1.4B image-text pairs       | 8B-24B         | Custom ViT                                   | Qwen-2-Instruct                                  |
| [Qwen2-VL](https://arxiv.org/pdf/2409.12191)                                                     | 2024 | Decoder-only   | 1B+ image-text pairs        | 7B-14B         | EVA-CLIP ViT-L                               | Qwen-2                                           |
| [Pixtral](https://arxiv.org/pdf/2410.07073)                                                      | 2024 | Decoder-only   | Undisclosed                 | 12B            | CLIP ViT-L/14                                | Mistral Large 2                                  |
| [LLaMA 3.2-vision](https://arxiv.org/pdf/2407.21783)                                            | 2024 | Decoder-only   | Undisclosed                 | 11B-90B        | CLIP                                         | LLaMA-3.1                                        |
| [Baichuan Ocean Mini](https://arxiv.org/pdf/2410.08565)                                          | 2024 | Decoder-only   | Image/Video/Audio/Text      | 7B             | CLIP ViT-L/14                                | Baichuan                                         |
| [TransFusion](https://arxiv.org/pdf/2408.11039)                                                  | 2024 | Encoder-decoder| Undisclosed                 | 7B             | VAE Encoder                                  | Pretrained from scratch on transformer architecture |
| [DeepSeek-VL2](https://arxiv.org/pdf/2412.10302)                                                 | 2024 | Decoder-only   | WiT, WikiHow                | 4.5B x 74      | SigLIP/SAMB                                  | DeepSeekMoE                                      |



##  2. <a name='Dataset'></a>🗂️ Dataset

- ⭐️ **ValueBench: Towards Comprehensively Evaluating Value Orientations and Understanding of Large Language Models**, ACL 2024, [[paper]](https://arxiv.org/abs/2406.04214), [[code]](https://github.com/Value4AI/ValueBench).
- https://lit.eecs.umich.edu/downloads.html
- **COMPO: Community Preferences for Language Model Personalization**, 2024.10, [[paper]](https://arxiv.org/pdf/2410.16027).
- **Cultural Commonsense Knowledge for Intercultural Dialogues**, CIKM 2024, [[paper]](https://dl.acm.org/doi/pdf/10.1145/3627673.3679768), [[dataset]](https://mango.mpi-inf.mpg.de/).

### Datasets and Evaluation for VLM 


| Benchmark Dataset                                        | Metric Type        |     Evaluation Method    | Source                 | Size (K) | Project 							|
|----------------------------------------------------------|:------------------:|:------------------------:|:-----------------------:|:---------:|:----------------------------------:|
| [MMTBench](https://arxiv.org/pdf/2404.16006)         	        | Multiple Choice    | Acc                      | AI Experts               | 30.1 | [Github Repo](https://github.com/tylin/coco-caption)|
| [MM-Vet](https://arxiv.org/pdf/2308.02490)					| LLM Eval           | Acc                      | Human                    | 0.2  | [Github Repo](https://github.com/yuweihao/MM-Vet) |
| [MM-En/CN](https://arxiv.org/pdf/2307.06281) 					 | Multiple Choice    | Acc                      | Human                    | 3.2   |[Github Repo](https://github.com/open-compass/VLMEvalKit)|
| [GQA](https://arxiv.org/abs/2305.13245)						| Answer Matching	| Acc, Consistency, Validity | Seed with Synthetic | 22,000 |[Website](https://cs.stanford.edu/people/dorarad/gqa/index.html)|
| [VCR](https://arxiv.org/abs/1811.10830)						| Multiple Choice    | Acc    					| MTurks                  | 290| [Website](https://visualcommonsense.com/)|
| [VQAv2](https://arxiv.org/pdf/1505.00468)						| Yes/No <br> Answer Matching | Acc, F1  			| MTurks                  | 1,100| [Github Repo](https://github.com/salesforce/LAVIS/blob/main/dataset_card/vqav2.md)|
| [MMMU](https://arxiv.org/pdf/2311.16502)						| Answer Matching; Multiple Choice				| Acc    | College Students        | 11.5 |[Website](https://mmmu-benchmark.github.io/) |
| [SEEDBench](https://arxiv.org/pdf/2307.16125)					| Multiple Choice    | Acc                      | Synthetic                | 19 |[Github Repo](https://github.com/AILab-CVC/SEED-Bench) |
| [RealWorld QA](https://x.ai/blog/grok-1.5v)					| Multiple Choice    | Acc                      | Human                    | 0.765|[Huggingface](https://huggingface.co/datasets/visheratin/realworldqa)|
| [MMMU-Pro](https://arxiv.org/pdf/2409.02813)					| Multiple Choice    | Acc                      | Human                    | 3.64  |[Website](https://mmmu-benchmark.github.io/#leaderboard)|
| [DPG-Bench](https://arxiv.org/pdf/2403.05135)					| Semantic Alignment | Alignment Score          | Synthetic                | 1.06   |[Website](https://ella-diffusion.github.io)|
| [MSCOCO-30K](https://arxiv.org/pdf/1405.0312)					| BLEU, Rouge, Similarity | Similarity Score      | MTurks                  | 30 |[Website](https://cocodataset.org/#home)|
| [TextVQA](https://arxiv.org/pdf/1904.08920)					| Answer Matching    | Acc                      | CrowdSource              | 45		|[Github Repo](https://github.com/facebookresearch/mmf)|
| [DocVQA](https://arxiv.org/pdf/2007.00398)					| Answer Matching    | Acc                      | CrowdSource              | 50 |[Website](https://www.docvqa.org/)|
| [CMMLU](https://arxiv.org/pdf/2306.09212)						| Multiple Choice    | Acc                      | College Students         | 11.5|[Github Repo](https://github.com/haonan-li/CMMLU)|
| [C-Eval](https://arxiv.org/pdf/2305.08322)					| Multiple Choice    | Acc                      | Human                    | 13.9|[Website](https://cevalbenchmark.com/)|
| [TextVQA](https://arxiv.org/pdf/1904.08920)					| Answer Matching    | Acc                      | Expert Human             | 28.6	|[Github Repo](https://github.com/facebookresearch/mmf)|
| [MathVista](https://arxiv.org/pdf/2310.02255)					| Answer Matching <br> Multiple Choice | Acc              | Human                    | 6.15  |[Website](https://mathvista.github.io/)|
| [MathVision](https://arxiv.org/pdf/2402.14804) 				| Answer Matching <br> Multiple Choice | Acc              | College Students         | 3.04 |[Website](https://mathvision-cuhk.github.io/)|
| [OCRBench](https://arxiv.org/pdf/2305.07895)					| Answer Matching (ANLS) | Acc                   | Human                    | 1 |[Github Repo](https://github.com/Yuliang-Liu/MultimodalOCR)|
| [MME](https://arxiv.org/pdf/2306.13394)						| Yes/No             | Acc                      | Human                    | 2.8 |[Github Repo](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation)|
| [InfographicVQA](https://arxiv.org/pdf/2104.12756) 			| Answer Matching    | Acc                      | CrowdSource              | 30|[Website](https://www.docvqa.org/)|
| [AI2D](https://arxiv.org/pdf/1603.07396)						| Answer Matching    | Acc                      | CrowdSource              | 1 |[Website](https://ai2d.med.upenn.edu/)|
| [ChartQA](https://arxiv.org/abs/2203.10244)					| Answer Matching    | Acc                      | CrowdSource/Synthetic    | 32.7 |[Github Repo](https://github.com/vis-nlp/ChartQA)|
| [GenEval](https://arxiv.org/pdf/2310.11513)					| CLIPScore <br> GenEval  | Similarity Score         | MTurks                  | 1.2	|[Github Repo](https://github.com/djghosh13/geneval)|
| [T2I-CompBench](https://arxiv.org/pdf/2307.06350)				| Multiple Metrics   | Similarity Score         | Synthetic                | 6 |[Website](https://karine-h.github.io/T2I-CompBench/)|
| [HallusionBench](https://arxiv.org/pdf/2310.14566)			| Yes/No             | Acc                      | Human                    | 1.13|[Github Repo](https://github.com/tianyi-lab/HallusionBench)|
| [POPE](https://arxiv.org/pdf/2305.10355)						| Yes/No             | Acc/Precision/Recall/F1  | Human                    | 9 |[Github Repo](https://github.com/RUCAIBox/POPE)|
| [MMLU](https://arxiv.org/pdf/2009.03300)						| Multiple Choice    | Acc                      | Human                    | 15.9	|[Github Repo](https://github.com/hendrycks/test)|
| [MMStar](https://arxiv.org/pdf/2403.20330)					| Multiple Choice    | Acc                      | Human                    | 1.5 |[Website](https://mmstar-benchmark.github.io/)|
| [M3GIA](https://arxiv.org/pdf/2406.05343)						| Multiple Choice    | Acc                      | Human                    | 1.8 |[Huggingface](https://huggingface.co/datasets/Songweii/M3GIA)|
| [InternetAGIEval](https://arxiv.org/pdf/2304.06364)			| Multiple Choice <br> Answer Matching | Acc/F1      | Human                    | 8.06    |[Github Repo](https://github.com/ruixiangcui/AGIEval)|
| [EgoSchem](https://arxiv.org/pdf/2308.09126)					| Multiple Choice    | Acc                      | Synthetic/Human          | 5     |[Website](https://egoschema.github.io/)|
| [MVBench](https://arxiv.org/pdf/2311.17005)					| Multiple Choice    | Acc                      | Synthetic/Human          | 4    |[Github Repo](https://github.com/OpenGVLab/Ask-Anything)|
| [MLVU](https://arxiv.org/pdf/2406.04264) 						| Multiple Choice    | Acc                      | Synthetic/Human          | 2.6   |[Github Repo](https://github.com/JUNJIE99/MLVU)|
| [VideoMME](https://arxiv.org/pdf/2405.21075)  | Multiple Choice    | Acc                      | Experts                  | 2.7                   |[Website](https://video-mme.github.io/)|
| [Perception-Test](https://arxiv.org/pdf/2305.13786)			| Multiple Choice    | Acc                      | CrowdSource       | 11.6 |[Github Repo](https://github.com/google-deepmind/perception_test)|



##  4. <a name='Toolenhancement'></a>⚒️ Tool enhancement
- **AI can help humans find common ground in democratic deliberation**, 2024.10, Science, [[paper]](https://www.science.org/doi/10.1126/science.adq2852).
- **PsyDI: Towards a Personalized and Progressively In-depth Chatbot for Psychological Measurements**, 2024, [[paper]](https://arxiv.org/abs/2408.03337), [[code]](https://github.com/opendilab/PsyDI).
- **ChatFive: Enhancing User Experience in Likert Scale Personality Test through Interactive Conversation with LLM Agents**, CUI 2024, [[paper]](https://dl.acm.org/doi/abs/10.1145/3640794.3665572)
- **LLM Agents for Psychology: A Study on Gamified Assessments**, 2024.02, [[paper]](https://arxiv.org/abs/2402.12326).
- **Generative Social Choice**, 2023.09, [[paper]](https://arxiv.org/abs/2309.01291).


###	4.1.  <a name='EmbodiedVLMagents'></a>Embodied VLM agents
- **VQA: Visual Question Answering**, 2016, [[paper]](https://arxiv.org/pdf/1505.00468).
- **Aligning Cyber Space with Physical World: A Comprehensive Survey on Embodied AI**, 2024, [[paper]](https://arxiv.org/pdf/2407.06886v1).
- **ChartLlama: A Multimodal LLM for Chart Understanding and Generation**, 2023, [[paper]](https://arxiv.org/pdf/2311.16483).
- **SciDoc2Diagrammer-MAF: Towards Generation of Scientific Diagrams from Documents guided by Multi-Aspect Feedback Refinement**, 2024, [[paper]](https://arxiv.org/pdf/2409.19242).

### 4.2. <a name='GenerativeVisualMediaApplications'></a>Generative Visual Media Applications

### 4.3. <a name='RoboticsandEmbodiedAI'></a>Robotics and Embodied AI

### 4.4. <a name='Human-CenteredAI'></a>Human-Centered AI

##  5. <a name='Alignment'></a>⛑️ Alignment

- **Aligning Large Language Models with Human Opinions through Persona Selection and Value–Belief–Norm Reasoning**, 2024.11, [[paper]](https://arxiv.org/pdf/2311.08385).
- **SafetyAnalyst: Interpretable, transparent, and steerable LLM safety moderation**, 2024.10, [[paper]](https://arxiv.org/abs/2410.16665).
- **Moral Alignment for LLM Agents**, 2024.10, [[paper]](https://arxiv.org/abs/2410.01639).
- **ProgressGym: Alignment with a Millennium of Moral Progress**, NeurIPS 2024 D&B Tract Spotlight, [[paper]](https://arxiv.org/abs/2406.20087), [[code]](https://github.com/PKU-Alignment/ProgressGym).
- **Strong and weak alignment of large language models with human values**, 2024.08, Nature Scientific Reports, [[paper]](https://www.nature.com/articles/s41598-024-70031-3).
- **STELA: a community-centred approach to norm elicitation for AI alignment**, 2024.03, Nature Scientific Reports, [[paper]](https://www.nature.com/articles/s41598-024-56648-4).
- **A Roadmap to Pluralistic Alignment**, ICML 2024, [[paper]](https://arxiv.org/abs/2402.05070), [[code]](https://github.com/jfisher52/AI_Pluralistic_Alignment).
- [*Value*] **What are human values, and how do we align AI to them?**, 2024.04, [[paper]](https://arxiv.org/abs/2404.10636).
- **Agent Alignment in Evolving Social Norms**, 2024.01, [[paper]](https://arxiv.org/abs/2401.04620).
- [*Norm*] **Align on the Fly: Adapting Chatbot Behavior to Established Norms**, 2023.12, [[paper]](https://arxiv.org/abs/2312.15907), [[code]](https://github.com/GAIR-NLP/OPO).
- [*MBTI*] **Machine Mindset: An MBTI Exploration of Large Language Models**, 2023.12, [[paper]](https://arxiv.org/abs/2312.12999), [[code]](https://github.com/PKU-YuanGroup/Machine-Mindset).
- **Training Socially Aligned Language Models in Simulated Human Society**, 2023, [[paper]](https://arxiv.org/abs/2305.16960), [[code]](https://github.com/agi-templar/Stable-Alignment).
- **Fine-tuning language models to find agreement among humans with diverse preferences**, 2022, [[paper]](https://arxiv.org/abs/2211.15006).
- **ValueNet: A New Dataset for Human Value Driven Dialogue System**, AAAI 2022, [[paper]](https://ojs.aaai.org/index.php/AAAI/article/view/21368), [[dataset]](https://liang-qiu.github.io/ValueNet/).

###  5.1. <a name='PluralisticAlignment'></a>🌈 Pluralistic Alignment
- [*Benchmark*] **Benchmarking Distributional Alignment of Large Language Models**, 2024.11, [[paper]](https://arxiv.org/abs/2411.05403).
- **Legal Theory for Pluralistic Alignment**, 2024.10, [[paper]](https://arxiv.org/abs/2410.17271).
- **Navigating the Cultural Kaleidoscope: A Hitchhiker’s Guide to Sensitivity in Large Language Models**, 2024.10, [[paper]](https://arxiv.org/abs/2410.12880), [[code and data]](https://github.com/NeuralSentinel/CulturalKaleidoscope).
- **PAD: Personalized Alignment at Decoding-Time**, 2024.10, [[paper]](https://arxiv.org/abs/2410.04070).
- **Policy Prototyping for LLMs: Pluralistic Alignment via Interactive and Collaborative Policymaking**, 2024.09, [[paper]](https://arxiv.org/abs/2409.08622).
- **Modular Pluralism: Pluralistic Alignment via Multi-LLM Collaboration**, 2024.06, [[paper]](https://arxiv.org/abs/2406.15951).

##  6. <a name='Simulation'></a>🚀 Simulation
- **OASIS: Open Agents Social Interaction Simulations on One Million Agents**, 2024.11, [[paper]](https://arxiv.org/abs/2411.11581), [[code]](https://github.com/camel-ai/oasis).
- **Generative Agent Simulations of 1,000 People**, 2024.11, [[paper]](https://arxiv.org/abs/2411.10109).
- **Social Science Meets LLMs: How Reliable Are Large Language Models in Social Simulations?**, 2024.11, [[paper]](https://arxiv.org/abs/2410.23426).
- **Multi-expert Prompting Improves Reliability, Safety, and Usefulness of Large Language Models**, EMNLP 2024, [[paper]](https://arxiv.org/pdf/2411.00492).
- **Simulating Opinion Dynamics with Networks of LLM-based Agents**, NAACL Findings 2024, [[paper]](https://aclanthology.org/2024.findings-naacl.211.pdf) [[code]](https://github.com/yunshiuan/llm-agent-opinion-dynamics)
- **Beyond demographics: Aligning role-playing llm-based agents using human belief networks**, EMNLP Findings 2024, [[paper]](https://arxiv.org/pdf/2406.17232)
- **The Wisdom of Partisan Crowds: Comparing Collective Intelligence in Humans and LLM-based Agents**, CogSci 2024, [[paper]](https://escholarship.org/content/qt3k67x8s5/qt3k67x8s5_noSplash_f34c019b5fef5ecab5b70e30108f787c.pdf)
- **Large Language Models can Achieve Social Balance**, 2024.10, [[paper]](https://arxiv.org/abs/2410.04054).
- **On the limits of agency in agent-based models**, 2024.09, [[paper]](https://arxiv.org/abs/2409.10568), [[code]](https://github.com/AgentTorch/AgentTorch).
- **United in Diversity? Contextual Biases in LLM-Based Predictions of the 2024 European Parliament Elections**, 2024.09, [[paper]](https://arxiv.org/abs/2409.09045).
- **Out of One, Many: Using Language Models to Simulate Human Samples**, 2022, [[paper]](https://arxiv.org/abs/2209.06899).
- **Social Simulacra: Creating Populated Prototypes for Social Computing Systems**, 2022, [[paper]](https://dl.acm.org/doi/abs/10.1145/3526113.3545616).
- **Generative Agents: Interactive Simulacra of Human Behavior**, 2023, [[paper]](https://arxiv.org/abs/2304.03442), [[code]](https://github.com/joonspk-research/generative_agents).
- **Using Large Language Models to Simulate Multiple Humans and Replicate Human Subject Studies**, 2023, [[paper]](https://proceedings.mlr.press/v202/aher23a.html), [[code]](https://github.com/GatiAher/Using-Large-Language-Models-to-Replicate-Human-Subject-Studies).
- **Large Language Models as Simulated Economic Agents: What Can We Learn from Homo Silicus?**, 2023 [[paper]](https://www.nber.org/papers/w31122), [[code]](https://github.com/johnjosephhorton/homo_silicus).
- **$S^3$: Social-network Simulation System with Large Language Model-Empowered Agents**, 2023, [[paper]](https://arxiv.org/abs/2307.14984).
- **Rethinking the Buyer’s Inspection Paradox in Information Markets with Language Agents**, 2023, [[paper]](https://openreview.net/forum?id=6werMQy1uz).
- **SocioDojo: Building Lifelong Analytical Agents with Real-world Text and Time Series**, 2023, [[paper]](https://openreview.net/forum?id=s9z0HzWJJp).
- **Humanoid Agents: Platform for Simulating Human-like Generative Agents**, 2023, [[paper]](https://arxiv.org/abs/2310.05418), [[code]](https://github.com/HumanoidAgents/HumanoidAgents).
- **When Large Language Model based Agent Meets User Behavior Analysis: A Novel User Simulation Paradigm**, 2023, [[paper]](https://arxiv.org/abs/2306.02552), [[code]](https://github.com/RUC-GSAI/YuLan-Rec).
- **Large Language Model-Empowered Agents for Simulating Macroeconomic Activities**, 2023, [[paper]](https://arxiv.org/abs/2310.10436).
- **Generative Agent-Based Modeling: Unveiling Social System Dynamics through Coupling Mechanistic Models with Generative Artificial Intelligence**, 2023, [[paper]](https://arxiv.org/abs/2309.11456).
- **Using Imperfect Surrogates for Downstream Inference: Design-based Supervised Learning for Social Science Applications of Large Language Models**, 2023.06, NeurIPS 2023, [[paper]](https://arxiv.org/abs/2306.04746).
- **Epidemic Modeling with Generative Agents**, 2023.07, [[paper]](https://arxiv.org/abs/2307.04986), [[code]](https://github.com/bear96/GABM-Epidemic).
- **Emergent analogical reasoning in large language models**, 2023.08, nature human behavior, [[paper]](https://www.nature.com/articles/s41562-023-01659-w).
- **MetaAgents: Simulating Interactions of Human Behaviors for LLM-based Task-oriented Coordination via Collaborative Generative Agents**, 2023.10, [[paper]](https://arxiv.org/abs/2310.06500).
- **War and Peace (WarAgent): Large Language Model-based Multi-Agent Simulation of World Wars**, 2023.11, [[paper]](https://arxiv.org/abs/2311.17227), [[code]](https://github.com/agiresearch/WarAgent).
- **Emergence of Social Norms in Large Language Model-based Agent Societies**, 2024.03, [[paper]](https://arxiv.org/abs/2403.08251), [[code]](https://github.com/sxswz213/CRSEC).
- **Large Content And Behavior Models To Understand, Simulate, And Optimize Content And Behavior**, ICLR-2024, [[paper]](https://openreview.net/forum?id=TrKq4Wlwcz)

##  7. <a name='Perspective'></a>👁️‍🗨️ Perspective

- **The benefits, risks and bounds of personalizing the alignment of large language models to individuals**, 2024.04, Nature Machine Intelligence, [[paper]](https://www.nature.com/articles/s42256-024-00820-y).
- **A social path to human-like artificial intelligence**, 2023.11, Nature Machine Intelligence, [[paper]](https://www.nature.com/articles/s42256-023-00754-x).
- **Using large language models in psychology**, 2023.10, Nature reviews psychology, [[paper]](https://www.nature.com/articles/s44159-023-00241-5).
