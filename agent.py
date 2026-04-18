import os
import chromadb
from dotenv import load_dotenv
from typing import TypedDict, List
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

# CHANGE 1: Define the State at the very top (Left-aligned)
class CapstoneState(TypedDict):
    question:      str          
    messages:      List[dict]   
    route:         str          
    retrieved:     str          
    sources:       List[str]    
    tool_result:   str          
    search_results: str         
    answer:        str          
    faithfulness:  float        
    eval_retries:  int          

# CHANGE 2: Wrap everything in the build_agent() function
def build_agent():
    llm  = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    client = chromadb.Client()
    
    # CHANGE 3: Fixed the try/except indentation
    try: 
        client.delete_collection("capstone_kb")
    except: 
        pass
    
    collection = client.create_collection("capstone_kb")

    DOCUMENTS = [
        {
            "id": "doc_001",
            "topic": "ResNet Residual Blocks",
            "text": """ResNet (Residual Network) introduced the concept of residual blocks to solve the vanishing gradient problem in extraordinarily
                deep neural networks. In a standard feedforward network, the output of a layer is passed directly to the next. In a residual block,
                a 'skip connection' or 'shortcut' bypasses one or more layers. This allows the network to learn identity functions easily. If the
                optimal mapping is closer to an identity mapping than to a zero mapping, it is easier for the solver to find the perturbations with
                reference to an identity mapping. This innovation allowed networks to scale to 152 layers and beyond, winning the 2015 ImageNet competition."""
        },
        {
            "id": "doc_002",
            "topic": "ResNet Bottleneck Architecture",
            "text": """In deeper ResNet variants like ResNet-50, ResNet-101, and ResNet-152, 'bottleneck' blocks are used to reduce computational
                complexity. Instead of two 3x3 convolutions used in ResNet-34, a bottleneck block uses three layers: 1x1, 3x3, and 1x1 convolutions.
                The first 1x1 layer reduces the dimensions (the bottleneck), the 3x3 layer processes it, and the final 1x1 layer restores the dimensions.
                This architecture significantly reduces the number of parameters and matrix multiplications, allowing for much deeper networks without a
                proportional increase in computational cost."""
        },
        {
            "id": "doc_003",
            "topic": "MobileNetV1 Depthwise Separable Convolutions",
            "text": """MobileNet architectures are designed specifically for efficient execution on mobile and embedded vision applications. The core building
            block of the original MobileNet is the depthwise separable convolution. Standard convolutions perform channel-wise and spatial-wise computations
                in one step. Depthwise separable convolutions split this into two separate layers: a depthwise convolution that applies a single spatial filter
                to each input channel, followed by a 1x1 pointwise convolution that linearly combines the outputs. This factorization drastically reduces both
                    computational cost and model size compared to standard convolutions."""
        },
        {
            "id": "doc_004",
            "topic": "MobileNetV2 Inverted Residuals",
            "text": """MobileNetV2 improves upon its predecessor by introducing inverted residual blocks with linear bottlenecks. Unlike classical ResNet blocks
            that have wide layers outside and a narrow bottleneck inside, MobileNetV2 expands the input representation to a higher dimension, filters it
                with a lightweight depthwise convolution, and then projects it back to a low-dimensional representation using a linear convolution. The skip
                connection connects the narrow bottlenecks. Using a linear activation function at the bottleneck prevents the loss of information that
                    would otherwise occur with non-linearities like ReLU in low-dimensional spaces."""
        },
        {
            "id": "doc_005",
            "topic": "Swin Transformer Shifted Windows",
            "text": """The Swin Transformer (Shifted Window Transformer) adapts the self-attention mechanism of standard Vision Transformers (ViTs) to
            make it computationally tractable for dense prediction tasks like object detection. Standard ViTs compute global self-attention across all
                patches, resulting in quadratic complexity with respect to image size. Swin Transformers compute self-attention locally within
                non-overlapping windows. To enable cross-window connections, the window partitioning is 'shifted' by half a window size between
                consecutive self-attention layers, allowing information to flow across boundaries while maintaining linear computational complexity."""
        },
        {
            "id": "doc_006",
            "topic": "Swin Transformer Patch Merging",
            "text": """A defining feature of the Swin Transformer is its hierarchical architecture, built via Patch Merging layers. As the network deepens,
            the number of tokens is reduced by merging neighboring patches. For example, a 2x2 group of neighboring patches is concatenated, and a linear
            layer is applied to project the feature dimension. This creates hierarchical feature maps similar to those in traditional convolutional neural
            networks (like ResNet). This multi-scale representation makes Swin Transformers highly effective for tasks requiring fine-grained spatial
            resolution, such as semantic segmentation."""
        },
        {
            "id": "doc_007",
            "topic": "Fine-Tuning Strategies",
            "text": """Fine-tuning pre-trained image classification models involves taking a model trained on a massive dataset (like ImageNet) and adapting
            it to a specific, often smaller, dataset. The standard practice is to freeze the weights of the early convolutional or attention layers, which
            capture generic features like edges and textures, and only update the weights of the final classification head and the deepest layers. This
            prevents overfitting on small datasets. A smaller learning rate is typically used during fine-tuning to avoid catastrophically forgetting the
            pre-trained weights."""
        },
        {
            "id": "doc_008",
            "topic": "Vision Transformers (ViT) vs CNNs",
            "text": """Vision Transformers (ViT) process images fundamentally differently than Convolutional Neural Networks (CNNs). Instead of using
            pixel arrays and sliding convolutional kernels, ViTs split an image into a grid of fixed-size patches (e.g., 16x16 pixels). Each patch is
            linearly embedded into a flat vector, appended with a positional encoding to retain spatial awareness, and then passed through a standard
            Transformer encoder. Because ViTs lack the strong inductive biases of CNNs (like translation invariance and local neighborhood focus),
            they generally require significantly larger datasets to train from scratch."""
        },
        {
            "id": "doc_009",
            "topic": "Parameter Counts and FLOPs",
            "text": """When comparing computational efficiency for deployment, it is vital to evaluate both Parameter Count and Floating Point Operations
            per Second (FLOPs). A standard ResNet-50 has roughly 25.6 million parameters and requires about 4.1 billion FLOPs for a single 224x224 image
            inference. In contrast, MobileNetV2 contains only 3.4 million parameters and requires a mere 300 million FLOPs. This massive reduction is
            what allows MobileNetV2 to achieve real-time frame rates on constrained edge devices while only sacrificing a few percentage points of top-1
            accuracy compared to ResNet-50."""
        },
        {
            "id": "doc_010",
            "topic": "Hardware Memory Constraints in Training",
            "text": """When adapting large architectures to hardware with strict memory limits, batch sizes often need to be drastically reduced to
            prevent Out Of Memory (OOM) errors. To compensate for small batch sizes, Gradient Accumulation is used: gradients are computed and summed
            over several sequential forward and backward passes before the optimizer updates the weights. Additionally, Mixed Precision Training
            (utilizing FP16 instead of FP32) can nearly halve the memory footprint of the model activations and weights while speeding up training
            on compatible tensor cores."""
        },
        {
            "id": "doc_011",
            "topic": "EfficientNet Compound Scaling",
            "text": """EfficientNet introduced a systematic way to scale up Convolutional Neural Networks to achieve better accuracy while maintaining
            efficiency. Prior networks scaled depth (number of layers), width (number of channels), or resolution independently, which often led to
            diminishing returns. EfficientNet's core innovation is 'Compound Scaling', which uniformly scales all three dimensions—width, depth, and
            resolution—using a fixed set of scaling coefficients determined by a grid search. This mathematically balanced approach led EfficientNet-B7
            to achieve state-of-the-art ImageNet accuracy while being 8.4x smaller than competing models."""
        },
        {
            "id": "doc_012",
            "topic": "Optimization: AdamW vs SGD",
            "text": """The choice of optimizer heavily influences model convergence. Stochastic Gradient Descent (SGD) with momentum is traditionally
            favored for training CNNs like ResNet from scratch because it often generalizes better to unseen data. However, Adam (Adaptive Moment 
            Estimation) is preferred for Transformers due to its per-parameter learning rates, which handle sparse gradients well. A major improvement 
            for Transformers was AdamW, which decouples weight decay from the gradient update. This prevents the regularization penalty from interfering 
            with the adaptive learning rates, leading to vastly improved performance for Vision Transformers."""
        },
        {
            "id": "doc_013",
            "topic": "Learning Rate Schedulers: Cosine Annealing",
            "text": """Static learning rates often cause models to plateau in sub-optimal local minima. Cosine Annealing is a popular scheduling technique 
            where the learning rate starts high and is gradually decreased following a cosine curve. The high initial rate allows the model to escape local 
            minima and traverse the loss landscape quickly, while the slow decay at the end allows the model to fine-tune its parameters and settle into a 
            flatter, more robust minimum. Sometimes, warm restarts are added (Cosine Annealing with Warm Restarts) to periodically pop the model out of local 
            minima."""
        },
        {
            "id": "doc_014",
            "topic": "Batch Normalization",
            "text": """Batch Normalization (BatchNorm) stabilizes and accelerates the training of deep CNNs by normalizing the inputs of each layer across 
            the mini-batch. It subtracts the batch mean and divides by the batch standard deviation, effectively maintaining the activations in a controlled 
            range. This reduces internal covariate shift, allowing for much higher learning rates and reducing the reliance on careful weight initialization. 
            During inference, BatchNorm uses running averages of the mean and variance computed during training, rather than the statistics of the inference 
            batch."""
        },
        {
            "id": "doc_015",
            "topic": "Layer Normalization in Transformers",
            "text": """While Batch Normalization is ubiquitous in CNNs, it struggles when batch sizes are very small or sequences vary in length. Layer 
            Normalization (LayerNorm) solves this by normalizing across the feature dimension for each individual token or instance independently of the 
            batch. This is why LayerNorm is the standard choice for Transformer architectures, including Vision Transformers. By normalizing the features 
            of each embedded patch separately, LayerNorm ensures stable gradient flow regardless of the hardware-constrained batch size."""
        },
        {
            "id": "doc_016",
            "topic": "Advanced Data Augmentation: MixUp",
            "text": """MixUp is an advanced data augmentation technique that encourages the model to behave linearly in-between training examples. Instead
            of feeding single images into the network, MixUp takes two random images and blends them together using a linear combination (e.g., 60% of
                Image A superimposed on 40% of Image B). The labels are also blended accordingly (e.g., 0.6 Cat + 0.4 Dog). This forces the network to 
                output smooth probability distributions rather than overconfident predictions, dramatically reducing overfitting and improving robustness 
                to adversarial examples."""
        },
        {
            "id": "doc_017",
            "topic": "Advanced Data Augmentation: CutMix",
            "text": """CutMix builds upon the concept of MixUp but preserves more local structural integrity, which is vital for CNNs. Instead of 
            transparently blending two images, CutMix cuts a rectangular patch from Image A and replaces it with a patch from Image B. The ground truth 
            labels are mixed proportionally to the area of the patches. By forcing the model to recognize objects from partial views and preventing it from 
            relying on a single discriminative feature (like a dog's face), CutMix improves localization ability and overall classification accuracy."""
        },
        {
            "id": "doc_018",
            "topic": "Regularization: Label Smoothing",
            "text": """Label Smoothing is a regularization technique designed to prevent a neural network from becoming overconfident. Standard Cross-Entropy
            loss uses 'hard' one-hot encoded targets (e.g., [1.0, 0.0, 0.0]). This encourages the model to push the logit of the correct class to infinity, 
            leading to overfitting and poor calibration. Label Smoothing softens these targets by distributing a small fraction of the probability mass among 
            the incorrect classes (e.g., [0.9, 0.05, 0.05]). This stabilizes training and helps the model generalize better to unseen data."""
        },
        {
            "id": "doc_019",
            "topic": "Top-1 vs Top-5 Accuracy",
            "text": """When evaluating image classification models on large datasets like ImageNet, two primary metrics are used: Top-1 and Top-5 accuracy.
            Top-1 accuracy strictly checks if the model's highest probability prediction matches the ground truth label. Top-5 accuracy is more forgiving; 
            it checks if the ground truth label is present anywhere within the model's top five highest probability predictions. Top-5 is often used for 
            datasets with many fine-grained categories (like 120 breeds of dogs) where even humans might struggle to distinguish the absolute top class."""
        },
        {
            "id": "doc_020",
            "topic": "Gradient Vanishing and Exploding",
            "text": """During backpropagation in deep networks, gradients are repeatedly multiplied by weight matrices. If the weights are small, the 
            gradients shrink exponentially, leading to the Vanishing Gradient problem where early layers stop learning. Conversely, if weights are large, 
            the gradients grow exponentially, causing the Exploding Gradient problem, resulting in numerical instability (NaN values). Architectures like 
            ResNet (via skip connections) and techniques like careful weight initialization (He/Xavier) and Normalization layers were specifically invented 
            to combat these issues."""
        },
        {
            "id": "doc_021",
            "topic": "Self-Attention Mechanism",
            "text": """The self-attention mechanism is the mathematical heart of the Transformer. For every element (or image patch) in a sequence, it 
            generates three vectors: Query, Key, and Value. To determine how much focus the current patch should place on all other patches, the model 
            calculates the dot product between the current patch's Query and the Keys of all patches. These scores are normalized using a softmax function 
            and then multiplied by the Value vectors. This allows the network to dynamically weigh the importance of long-range dependencies across the 
            entire image."""
        },
        {
            "id": "doc_022",
            "topic": "Multi-Head Attention",
            "text": """Multi-Head Attention expands on standard self-attention by running multiple attention mechanisms in parallel. Instead of computing one 
            set of attention weights, the network projects the Queries, Keys, and Values into several smaller-dimensional 'heads'. Each head learns to attend 
            to different types of relationships—for instance, one head might focus on spatial proximity, while another focuses on color contrast. The outputs 
            of all heads are concatenated and linearly transformed. This provides the model with a richer, more diverse set of feature representations."""
        },
        {
            "id": "doc_023",
            "topic": "Zero-Shot Classification via CLIP",
            "text": """CLIP (Contrastive Language-Image Pre-training) represents a paradigm shift from traditional image classification. Instead of 
            predicting a fixed set of classes, CLIP is trained to match images with their corresponding text descriptions using a contrastive loss function. 
            During inference, you can perform 'zero-shot' classification by providing the model with a new image and a list of text prompts (e.g., 'a photo 
            of a cat', 'a photo of a car'). The model computes the cosine similarity between the image embedding and the text embeddings, selecting the 
            prompt that matches best without requiring any task-specific fine-tuning."""
        },
        {
            "id": "doc_024",
            "topic": "Dropout and Weight Decay",
            "text": """Dropout and Weight Decay are the two most common regularization techniques to combat overfitting. Dropout works by randomly zeroing
            out a percentage of activations during training, forcing the network to learn redundant representations rather than relying heavily on a few 
            specific neurons. Weight Decay (L2 Regularization) adds a penalty to the loss function proportional to the squared magnitude of the model's 
            weights. This discourages the network from learning extremely large weight values, resulting in a smoother, more generalized decision boundary."""
        },
        {
            "id": "doc_025",
            "topic": "Gradient Clipping",
            "text": """Gradient Clipping is a technique used to ensure training stability, particularly in architectures prone to exploding gradients. Before
            the optimizer updates the model parameters, the norm (magnitude) of the gradient vector is calculated. If this norm exceeds a pre-defined 
            threshold, the entire gradient vector is scaled down proportionately so its norm exactly matches the threshold. This ensures the direction of 
            the gradient step remains the same, but the magnitude is capped, preventing catastrophic jumps in the loss landscape that could derail 
            training."""
        }
    ]
    
    texts = [d["text"] for d in DOCUMENTS]
    collection.add(documents=texts, embeddings=embedder.encode(texts).tolist(),
                   ids=[d["id"] for d in DOCUMENTS],
                   metadatas=[{"topic":d["topic"]} for d in DOCUMENTS])

    # ── Node 1: Memory ─────────────────────────────────────────
    def memory_node(state: CapstoneState) -> dict:
        msgs = state.get("messages", [])
        msgs = msgs + [{"role": "user", "content": state["question"]}]
        if len(msgs) > 6:  # sliding window: keep last 3 turns
            msgs = msgs[-6:]
        return {"messages": msgs}

    # ── Node 2: Router ─────────────────────────────────────────
    def router_node(state: CapstoneState) -> dict:
        question = state["question"]
        messages = state.get("messages", [])
        recent   = "; ".join(f"{m['role']}: {m['content'][:60]}" for m in messages[-3:-1]) or "none"

        prompt = f"""You are a router for a chatbot about Deep Learning, Neural Networks, and Computer Vision architectures.
        Available options:
        - retrieve: search the knowledge base for topics like ResNet, MobileNet, Swin Transformers, optimizers, and training strategies.
        - memory_only: answer from conversation history.
        - tool: use the web_search tool ONLY IF the user asks about a recent paper, framework version, or topic explicitly outside the core architectures.

        Recent conversation: {recent}
        Current question: {question}
        Reply with ONLY one word: retrieve / memory_only / tool"""

        response = llm.invoke(prompt)
        decision = response.content.strip().lower()

        if "memory" in decision:       decision = "memory_only"
        elif "tool" in decision:       decision = "tool"
        else:                          decision = "retrieve"

        return {"route": decision}

    # ── Node 3: Retrieval ──────────────────────────────────────
    def retrieval_node(state: CapstoneState) -> dict:
        q_emb   = embedder.encode([state["question"]]).tolist()
        results = collection.query(query_embeddings=q_emb, n_results=3)
        chunks  = results["documents"][0]
        topics  = [m["topic"] for m in results["metadatas"][0]]
        context = "\n\n---\n\n".join(f"[{topics[i]}]\n{chunks[i]}" for i in range(len(chunks)))
        return {"retrieved": context, "sources": topics}

    def skip_retrieval_node(state: CapstoneState) -> dict:
        return {"retrieved": "", "sources": []}

    # ── Node 4: Tool ───────────────────────────────────────────
    def tool_node(state: CapstoneState) -> dict:
        question = state["question"]
        print(f"  [tool] Searching web for: {question}")
        
        try:
            from ddgs import DDGS
            results = DDGS().text(question, max_results=3)
            
            if not results:
                tool_result = "Search returned no results."
            else:
                tool_result = "\n\n".join(f"Title: {r['title']}\nSnippet: {r['body'][:300]}" for r in results)
                
        except ImportError:
            tool_result = "Error: 'ddgs' library not installed. Please run: pip install ddgs"
        except Exception as e:
            tool_result = f"Search error: {str(e)}"

        return {"tool_result": tool_result, "search_results": tool_result}

    # ── Node 5: Answer ─────────────────────────────────────────
    def answer_node(state: CapstoneState) -> dict:
        question    = state["question"]
        retrieved   = state.get("retrieved", "")
        tool_result = state.get("tool_result", "")
        messages    = state.get("messages", [])
        eval_retries= state.get("eval_retries", 0)

        context_parts = []
        if retrieved:
            context_parts.append(f"KNOWLEDGE BASE:\n{retrieved}")
        if tool_result:
            context_parts.append(f"TOOL RESULT:\n{tool_result}")
        context = "\n\n".join(context_parts)

        if context:
            system_content = f"""You are an advanced Deep Learning Research Assistant.
        Answer the user's question using ONLY the provided academic context below.
        If the answer is not explicitly contained in the context, do not guess or hallucinate parameters. State clearly: "I don't have that information in my current knowledge base."
        

        {context}"""
        else:
            system_content = """You are a Deep Learning Research Assistant. Answer based on the conversation history."""

        if eval_retries > 0:
            system_content += "\n\nIMPORTANT: Your previous answer did not meet quality standards. Answer using ONLY information explicitly stated in the context above."

        lc_msgs = [SystemMessage(content=system_content)]
        
        for msg in messages[:-1]:
            lc_msgs.append(HumanMessage(content=msg["content"]) if msg["role"] == "user" else AIMessage(content=msg["content"]))
            
        lc_msgs.append(HumanMessage(content=question))

        response = llm.invoke(lc_msgs)
        return {"answer": response.content}

    # ── Node 6: Eval ───────────────────────────────────────────
    FAITHFULNESS_THRESHOLD = 0.7
    MAX_EVAL_RETRIES       = 2

    def eval_node(state: CapstoneState) -> dict:
        answer   = state.get("answer", "")
        context  = state.get("retrieved", "")[:500]
        retries  = state.get("eval_retries", 0)

        if not context:
            return {"faithfulness": 1.0, "eval_retries": retries + 1}

        prompt = f"""Rate faithfulness: does this answer use ONLY information from the context?
        Reply with ONLY a number between 0.0 and 1.0.
        1.0 = fully faithful. 0.5 = some hallucination. 0.0 = mostly hallucinated.

        Context: {context}
        Answer: {answer[:300]}"""

        result = llm.invoke(prompt).content.strip()
        try:
            score = float(result.split()[0].replace(",", "."))
            score = max(0.0, min(1.0, score))
        except:
            score = 0.5

        return {"faithfulness": score, "eval_retries": retries + 1}

    # ── Node 7: Save ───────────────────────────────────────────
    def save_node(state: CapstoneState) -> dict:
        messages = state.get("messages", [])
        messages = messages + [{"role": "assistant", "content": state["answer"]}]
        return {"messages": messages}

    # ── Routing functions ──────────────────────────────────────
    def route_decision(state: CapstoneState) -> str:
        route = state.get("route", "retrieve")
        if route == "tool":        return "tool"
        if route == "memory_only": return "skip"
        return "retrieve"

    def eval_decision(state: CapstoneState) -> str:
        score   = state.get("faithfulness", 1.0)
        retries = state.get("eval_retries", 0)
        if score >= FAITHFULNESS_THRESHOLD or retries >= MAX_EVAL_RETRIES:
            return "save"
        return "answer"  # retry

    # ── Build the graph ────────────────────────────────────────
    graph = StateGraph(CapstoneState)

    graph.add_node("memory",    memory_node)
    graph.add_node("router",    router_node)
    graph.add_node("retrieve",  retrieval_node)
    graph.add_node("skip",      skip_retrieval_node)
    graph.add_node("tool",      tool_node)
    graph.add_node("answer",    answer_node)
    graph.add_node("eval",      eval_node)
    graph.add_node("save",      save_node)

    graph.set_entry_point("memory")
    graph.add_edge("memory",   "router")

    graph.add_conditional_edges(
        "router", route_decision,
        {"retrieve": "retrieve", "skip": "skip", "tool": "tool"}
    )

    graph.add_edge("retrieve", "answer")
    graph.add_edge("skip",     "answer")
    graph.add_edge("tool",     "answer")

    graph.add_edge("answer", "eval")
    graph.add_conditional_edges(
        "eval", eval_decision,
        {"answer": "answer", "save": "save"}
    )
    graph.add_edge("save", END)

    checkpointer = MemorySaver()
    agent_app = graph.compile(checkpointer=checkpointer)

    return agent_app, embedder, collection