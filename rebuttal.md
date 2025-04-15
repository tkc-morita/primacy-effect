# Overview of the revisions

We first thank all four reviewers for their insightful and constructive comments.
This post summarizes the major revisions made on the paper, which address the common concerns raised by the reviewers.

1. Review on LLMs exhibiting the primacy effect

The Preliminaries section has been extended to include a discussion of related studies on LLMs exhibiting the primacy (moved from the "Comparison with Data/Task-Oriented Primacy Effects" subsection in the Discussion).
This revision helps clarify the novelty of our study: while LLMs studied in the previous works can inherit human's cognitive bias through natural language data, the SSM explored in our work was trained **from scratch** on a psychologically grounded memorization task.

2. Exploration of Transformer

We conducted an additional experiment on a (single-layered) Transformer model whose attention covered the whole input squence.
Unfortunately, however, the Transformer achieved perfect accuracy even when we increased the input length and vocabulary size to the maximum allowed by our computational resources.
Consequently, we were unable to identify a "memory bias" in the model (i.e., which part of the sequence it prioritizes when complete memorization is impossible).

This result is not surprising, given that Transformer's "memory" is essentially a complete, uncompressed copy of the input sequence.
For each verification query, the model can scan through all study items and check for a match.
More technically, the model can solve the task simply by computing the dot-product similarity between the (linearly projected) embeddings of the verification query and the study items.
This yields either a uniform or one-hot-like attention distribution, depending on the absense or existence of the queried study item.
Hence, the assessment of memory capacity in the proposed paradigm is largely irrelevant for Transformer, since its "memory" is neither compressed nor mixed across input tokens.
This analysis has been added to a new Appendix section.

# Dyuk

Thank you for your comments.
Please find our responses to your individual questions below.

> There does seem to be a primacy effect, but it's interesting that the performance on items later in the sequence is very low (even lower than the LSTM, which performs consistently across items from different points in the sequence). It seems like there could be a lot more exploration about where this effect comes from, and more analysis on the behavior of the network. For instance, are there any patterns in the false postives and false negatives that drive the primacy effect? The effect is also more dramatic depending on the length of the verification phase-- why is that?

We agree that the low accuracy of the SSM for the study items presented at the end and immediately verified (i.e., the bottom-left region of the heatmaps) represents another interesting property as a model of "long-term memory";
this result suggests that memorized information was not trievable immediately after the observation of the study items.

Regarding the analysis of the false negative and false positive patterns, Fig. 3 already presents both types of information.
The main square heatmaps display the proportion of true positives against false negatives (i.e., the recall scores), while the separate top row in each panel (labeled "Distr.") shows the proportion of true negatives vs. false positives.
These interpretations have been clarified in the revised manuscript (Line 333-338).

An interesting example of false negatives is the apparent "lag" of memory retrieval mentioned above, although we have not yet derived a mathematical explanation for this phenomenon.
By contrast, false positives exhibited only a slight decrease over verification time, which appears to align with gradual memory decay.

As for the effect of input length (and corresponding verification length), we are unsure in what sense you found the result to be "more dramatic".
We would greatly appreciate it if your could provide further clarification or elaboration on this point.
Our interpretation of the difference among Fig. 3B vs. F vs. G is simply the relation between the input length and the task difficulty.
Specifically, a shorter sequence (L=64) is easier to memorize, leading to higher overall accuracy, while a longer sequence (L=256) poses greater challenges for both memorization and retention.


> I understand that delta-T is an important hyperparameter, but I’m not sure what I learned from figure 4. The text says “The key factor responsible for the primacy effect in the SSM appears to be the time-step size, ∆t. After training on the memorization task, ∆t values concentrated below a specific threshold (∆t ≤ 0.03)”. Even if the learned delta-ts are at a certain range, I don't think that is sufficient to show that this variable is responsible for the primacy effect. Additionally, if the other delta-ts just result in a poorly performing network, that doesn't seem like a sufficient comparison to say that the delta-t hyperparameter is the reason for primacy in the network.

On the one hand, we acknowledge that the lack of a rigorous analysis of $\Delta t$ is a major limitation of the present study.
As noted in the Discussion section, adequately small values on $\Delta t$ are only a necessary condition for the primacy effect (or the retention of the initial study items), not the sufficient condition (Line 432-436).

On the other hand, our focus on $\Delta t$ emerged through a process of elimination.
Even after fixing the state matrix $A$, we continued to observe the primacy effect, leaving $\Delta t$ as the sole parameter capable of influencing the *dynamics* of the SSM.
In other words, all remaining parameters are confined to feedforward transforms.
This reasoning has been clarified in the revised manuscirpt (Line 374-377).
We also have also reported that freezing $\Delta t$ resulted in a complete faulure of learning (see Footnote 8).

> There were several parts of the "Finite Queue"/transformers introduction paragraphs that I felt could be modified or improved. The rolling queue explanation was unconvincing as a reason to exclude these models. There is a fundamental storage limit for any memory model, but it is still a fair question to ask whether, within the context of the available queue, a primacy effect is shown. It is also confusing since later in the Discussion "Comparison with Data/Task-Oriented Primacy Effects", the authors actually mention papers that show a primacy effect in transformers. It seems more useful to just acknowledge the limitation of the present study (e.g., that they don't test transformers) and cite the other papers earlier.

Following your suggestion, we made the following revisions:
1. We revised the "Finite-Queue/Markov Models" section to discuss two possible cases;
   - i. Early observations fall outside the memory window, and
   - ii. Modern powerful models (e.g., Transformer) can cover most empirical sequences (newly added; Line 187-197).
2. We moved the review of LLM-related studies from the Discussions section to the Preliminaries (Line 198-241). We also clarified the novelty of our study in contrast to these prior LLM works.
3. We conducted an additional experiment on Transformer. However, we were unable to reach its memory capacity, which is necesary to diagnose its bias pattern (Line 303-315). In a new Appendix section, we provide a theoretical explanation for why Transformer can trivially solve the adopted task (Line 892-919).

> There were several parts of the discussion "Implications for Computational Neuroscience" that felt weak to me. There was several points about how spiking versions of SSMs exist, and that this thus means the SSMs are a more biologically grounded network than other models: "Given that spiking neural networks remain the most plausible model of biological neuronal activity to date, a spiking version of the SSM studied here could serve as a new neurobiological model of the primacy effect." However, many networks can be converted to a spiking counterpart, and other neural networks have also been proposed that can demonstrate the primacy effect (e.g. Wong, Kahn, and Sherrington 1991). Thus, these arguments seem like an overreach to me. Furthermore, I think this model can be of interest to computational neuroscientists without invoking a spiking model. It's reasonable to ask what high-level computations result in a primacy effect in a model, and this can be useful to neuroscientists in and of itself (this also seems like the clear next step in this model, as it is not explored why the primacy effect is arising in the SSM).

We appreciate your references to existing studies on neural network analyses of the primacy effect.
The revised manuscript now acknowledges these prior works, and the novelty of our findings in comparison has been clarified (Line 503-519).
Specifically, we emphasize that the SSM was developed independently of those earlier models and, thus, presents a unique and complementary hypothesis for the mechanism underlying the biological primacy effect.

Meanwhile, we have retained the discussion on spiking neural networks, as it offers a counter-argument against the potential claim that the SSM lacks relevance to biological networks.
However, we have revised the phrasing to avoid making an affirmative justification of its biological plausibility based on the spiking instantiation (Line 520-543).


> In the same discussion section mentioned above, the authors write: "Classical theories on the primacy effect in human and animal memory hypothesize a dual-storage mechanism, attributing the primacy and recency effects to long- and short-term memory systems, respectively. By contrast, the findings of the present study may contribute to the development of a novel and more neurobiologically grounded model of the primacy effect." However, I don't think the classical theories are mutually exclusive with the observation that SSMs may show primacy effects. These may just be different scales of explanation. For instance, maybe the SSM implicitly has a dual-storage mechanism as well, and there are separate representation spaces in the SSM that act as a long-term or short-term memory store.

Thanks to the inclusion of additional references on neural network analyses of the primacy effect, we believe the revised manuscript no longer implies a mutually exclusive relationship between the SSM and dual-store model (Line 485-519).
On the contrary, the revised manuscript now explicitly acknowledges the absence of a recency effect in the SSM and, accordingly, highlights the need for an additional "short-term memory" module to account for it (Line 558-566).
In this light, the SSM appears better suited as a model of "long-term memory" within a dual-store framework, rather than a comprehensive model of biological memory.

> Minor: figure 4, not sure why you need to make channel and training runs separate. Maybe just average over the training runs.

A key difficulty in averaging across runs lies in aligning the $\Delta t$ values between runs; is it truly reasonable to sort the $\Delta t$ within each run and then compute the rank-wise averages?
We consider that both cross-run averaging and our approach of "flattening" samples from different runs for visualization are rooted in the same underlying statistical assumption that the $\Delta t$ values across different runs are drawn from the same distribution.
Accordingly, to support this assumption, we instead report the **variations** of the $\Delta t$ distributions across runs in a new Appendix section (Fig. 5).


# ATDs

Thank you for your comments.
Please find our responses to your individual questions below.

> INTEREST I am not completely sure if this has broad interest or is only relevant to a subdiscipline. How biologically plausible are structured space models?

Although the SSM was developed with the aim of advancing practically useful artificial intelligence, we consider that it also offers biologically meaningful insights.
As noted in Line 520-543, previous studies have modeled biological neuronal activity using the SSM framework.
Thus, the model is not merely a industrial tool, but also can also serve as a biologically relevant model for understanding cognitive phenomena.
We hope that our findings will encourage greater interest in the SSM from the cognitive neuroscience community.


> SOUNDNESS The experiments were adequate but I would have liked to see more attempts to explain the surprising results.

We do acknowledge that the limited mathematical understanding of our findings represents a major limitation of the present study.
Indeed, the SSM contains few factors that influence its latent dynamics.
Specifically, all its parameters aside from the state matrix $A$ and the time-step size $\Delta t$ are confined to feedforward transforms.
Thus, the persistence of the primacy effect even after fixing $A$ points to $\Delta t$ as the remaining influential factor, motivating our inspection of it in the "Distribution of the Time-Step Sizes" section.
We have added this rationale to the opening paragraph of the "Distribution of the Time-Step Sizes" section (Line 374-377).


# rege

Thank you for your comments.
Please find our responses to your individual questions below.

> 1 . Interpretation of the lack of recency effect: it is an intriguing feature that the SSM shows the primacy bias. This has been a challenge for most current ANN architectures attempting to model memory, which are generally better at showing the recency rather than the primacy bias. However, interestingly, the SSM in this study failed at showing the recency effect, and the manuscript did not provide much insight into why this was the case. The authors stated in the Discussion, without further notes, that "...the question remains as to why recent observations were remembered less accurately despite the exponentially decaying measure underlying the polynomial approximation theory." This is a critical question and it would be helpful to provide speculation on possible answers and/or strategies to finding the answer.

We do acknowledge that the limited mathematical understanding of both the emergence of the primacy effect and the absence of the recency effect represents a major limitation of the present study.
Given this limitation, we suspect that the time-step size, $\Delta t$, as the key parameter, **based on a process of elimination**.
Indeed, the SSM contains few factors that influence its latent dynamics.
Specifically, all its parameters aside from the state matrix $A$ and $\Delta t$ are confined to feedforward transforms.
Thus, the persistence of the primacy effect even after fixing $A$ points to $\Delta t$ as the remaining influential factor, motivating our inspection of it in the "Distribution of the Time-Step Sizes" section.
We have added this rationale to the opening paragraph of the "Distribution of the Time-Step Sizes" section (Line 374-377).


> One possibility that may be worth considering is the link to finite queue/Markov models (mentioned in Introduction). These models, as the authors also stated, are good at capturing recency but have theoretical limitations in modeling primacy. Given that Markov models are a special type of state-space models, the link between the SSM implemented in the current work and finite queue/Markov models can provide valuable insight into why the primacy/recency effect arise in one type of the models but not in the other. Relatedly, would the pros and cons of each model suggest a way to combine them so one could model both primacy and recency?Discussion along this line can provide useful information for improving the current model.

Thank you for highlighting the importance of integrating long- and short-term memory components.
Given the absence of a recency effect, we agree that the SSM is better positioned as a model of long-term memory within a dual-store framework, rather than a comprehensive model of biological memory.
The revised manuscript now emphasizes the importance of investigating joint learning between the SSM and a complementary module modeling the short-term memory (Line 558-566).

As you pointed out, finite queue/Markov models are the most straightforward candidates for capturing the recency effect.
This point is noted in the Preliminaries section (Line 180-184).

> 2 . Related to the point above, and given that SSMs are generative models, the recency effects in the current SSM implementation can be further tested using a free-recall paradigm. Specifically, one can use the trained SSM (essentially an RNN) to generate sequences of integers and test whether neighboring items are more likely to be recalled together compared to temporally distant items. This test would be complementary to the old/new accuracy test reported in the manuscript. Importantly, the transition patterns among recalled items would provide additional information for inferring the encoding and retrieval mechanisms of this SSM.

While it is true that the SSM is capable of generating non-binary sequences, the main challenge in investigating it under the free-recall paradigm lies in the **training of order-agnostic responses**.
Canonical sequence learning tasks (e.g. language modeling) rely on order-sensitive optimization (token-wise loss computation).
Although it is theoretically possible to design an order-agnostic objective function (using the framework of optimal transport; Cuturi, 2013), our initial explorations revealed that the model's performance remained suboptimal under this approach, yielding lower accuracy than in theoretically more demanding tasks that required order-sensitive reconstruction.
For this reason, we adopted the binary verification paradigm, which has also proven useful for studying non-human animals.
This rationale is dicussed in Footnote 5.

> 3 . The authors chose LSTM for comparison with the SSM, which does not seem to be the best benchmarking choice given that LSTM doesn't particularly concern biological plausibility or relevance to human behavior. There are recently developed ANN architectures based on biologically plausible functions (e.g. with exponential decay, Misty et al., 2025), or known to show primacy and recency biases (e.g. Guo et al., 2024; Liu et al., 2024), which all seem more relevant to this study.

We appreciate your references to related studies on the primacy effect.
We have added a new subsection under the Preliminaries that clarifies the differences and novelty of our study in comparison to these prior works (Line 198-241).
Specifically, the previous studies utilized LLMs trained on human-generated linguistic data; consequently, the observed primacy effects are attributed (by both the authors and ourselves) to  cognitive biases inherited from humans through the data.
By contrast, the SSM in our study was trained **from scratch** on a statistically balanced task that was designed following psychological experiments.
Hence, our findings are attributed to the intrinsic properties of the SSM itself, rather than to biases present in the training data.

Because LLMs are trained on heterogeneous datasets and objectives, their direct comparison with the SSM/LSTM under our paradigm is not feasible.
Instead, we conducted an additional experiment on Transformer, training it from scratch on the memory verification task.
However, the model attained perfect accuracy, even after we increased both input length and vocabulary size to the upper bounds set by our computational capacity (Line 303-315).
Consequently, we were unable to examine its memory biases, which become visible only when suboptimal model performance is achieved.

Give this result, we provide a theoretical explanation in a new Appendix section outlining why Transformer can solve the memory verification task trivially.
Essentially, this task is ill-suited for testing Transformer because its "memory" consists of a lossless copy of the full input sequence.
This enables the model to scan for matches between study items and verification queries without any concern about memory compression or temporal decay.


> 4 . Figure 1 is missing the y-axis label.

Thank you for notifying us of this.
The figure now has a y-label "Kernel Value".


# ktWE

Thank you for your comments.
Please find our responses to your individual questions below.

> Interest:

> The primacy effect is of course specific to a subfield of the CCN field. The claim that the primacy effect is intrinsic to the SSM architecture is noteworthy, suggesting that at least a more rigorous examination is needed.

We are concerned that our previous draft may have conveyed a misleading impression, and we would like to begin by clarifying our core arguments.

When we state that "the observed primacy effect is attributed to the intrinsic properties of the SSM per se", our intention was to highlight a contrast with prior studies on LLMs, which were trained on human-generated linguistic data.
Specifically, the primacy effect exhibited by LLMs is attributed to human's cognitive biases embedded in the data (as acknowledged by the authors of those studies).
By contrast, our study trained the SSM **from scratch** on statistically balanced synthetic data, with a task design mirroring psychological experiments.
This controlled setup allows us to examine the model’s "intrinsic" memory biases, without being "contaminated" by human-induced biases present in naturalistic datasets.

**Crucially, we do NOT claim that the SSM is the ONLY neural network architecture capable of reproducing the primacy effect.**
Given the vast number of models proposed in the literature, it is entirely possible that other, non-SSM architectures also exhibit similar behavior.
Rather, the significance of our findings lies in the following two aspects:
1. The emergence of the primacy effect **contradicts the intended inductive bias of the SSM**, which is based on a polynomial approximation of observed signals under an exponentially decaying measure. (Thus, the anticipated bias is the recency effect.)
2. The observed primacy effect is **not attributable to inheritance of human-induced biases**, as the model was trained exclusively on synthetically constructed, statistically balanced data.

And to support these claims, we focused on systematically exploring the internal configurations of the SSM&mdash;such as learnable vs. frozen parameters, types of polynomial bases, and different initializations of $\Delta t$&mdash;rather than covering more advanced (and complex) alternative models (unlike the benchmark races towards the state-of-the-art).

These clarifications have been added in a new subsection titled "Data-Driven Primacy Effect in Language Models" in the Preliminaries (Line 198-241).

Now going back to the broadness of our potential audience, we believe that our findings are relevant and valuable to both the machine learning and cognitive neuroscience communities.
From the perspective of explnable AI (XAI), the emergence of a theoretically unexpected, non-human-rooted bias is particularly noteworthy.
From a neuroscientific standpoint, the SSM offers a novel (not exclusive) hypothesis for the mechanism underlying the primacy effect, developed independently of existing models that were specifically designed for replicating the cognitive bias.

Finally, we do acknowledge that the absence of a rigorous (mathematical) explanation for the observed primacy effect is the major limitation of our study.
Nevertheless, we have a rationale for focusing on the time-step size parameter, $\Delta t$.
The SSM contains few factors that influence its latent dynamics.
Specifically, all its parameters aside from the state matrix $A$ and $\Delta t$ are confined to feedforward transforms.
Thus, the persistence of the primacy effect even after fixing $A$ points to $\Delta t$ as the remaining influential factor, motivating our inspection of it in the "Distribution of the Time-Step Sizes" section.
We have added this rationale to the opening paragraph of the "Distribution of the Time-Step Sizes" section (Line 374-377).


> Soundness:

> An issue with the manuscript is that the findings seem pretty narrow, as the SSM is compared to very old-school LSTMs, rather than more modern RNNs like xLSTMs, where the latter specifically address some of the crucial shortcomings of LSTMs. Moreover, while it is true that Transformers do not explicitly maintain memory outside their so-called context-window, this context-window can be very long, up to a million tokens in for example Google's Gemini model. Inside these windows, the primacy effect has been observed extensively of course. As noted, the SSMs similarly have a memory capacity.

First and foremost, we wish to re-emphasize that our goal is NOT to claim that the SSM is the ONLY architecture capanble of exhibiting the primacy effect.
Rather, the presentation of LSTM results serves to demonstrate that **not all** neural networks exhibit the primacy effect.
LSTM was selected as a representative baseline in a sense that it has been studied extensively in the literature (thus, older is better).

Nevertheless, we fully agree that investigating additional architectures is valuable.
Accordingly, we conducted supplementary experiments with Transformer and xLSTM.
Unfortunately, however, both the models achieved perfect accuracy even when we increased the input length and vocabulary size to the maximum allowed by our computational resources.
Consequently, we were unable to identify a "memory bias" in the models (i.e., which part of the sequence they prioritize when complete memorization is impossible).
These results are reported in Line 303-315 (Tranformer) and Footnote 7 (xLSTM) of the revised manuscript.

Additionally, we have added a new Appendix section that provides a theoretical explanation of why Transformer can solve the memory verification task trivially.
Essentially, this task is ill-suited for testing Transformer because its "memory" consists of a lossless copy of the full input sequence.
This enables the model to scan for matches between study items and verification queries without any concern about memory compression or temporal decay.


> It is not so clear whether the evidence supports the claims. E.g., do modern RNNs like xLSTMs exhibit the primacy effect? And since SSMs can also be described in a fashion close to (linear) Transformers, wouldnt the observation that Transformers exhibit the primacy effect suggest that this would also likely be the case for modern RNNs? The lack of recency effect is also odd - it is typically found in Transformers.

As already explained in our response to your comments on "Interest", what we meant by "the observed primacy effect is attributed to the intrinsic properties of the SSM per se" was NOT intended as a comparison with alternative neural networks.
Instead, our claim is situated in contrast to prior studies on LLMs, which were trained on natural language data and thus can inherit human biases.
(We interpret the Transformer-based studies you refer to regarding the primacy effect as falling into this category of LLM research.)

From this perspective, we believe that our experiment provides concrete evidence that the SSM can exhibit the primacy effect **in the absence of human-originated biases**, even though its underlying mathematical mechanism remains open.


> The study demonstrates that the primacy effect is stronger in the SSM for smaller initialization values of Delta t.

Comparing Figure 3B ($10^{-4} \leq \Delta t 10^{-1}$) and E ($10^{-4} \leq \Delta t 10^{0}$), we observe only a slight decrease in accuracy over verification time (left to right) in the top rows of the latter (where the primacy effect is represented).
If this is what you are referring to, our suspicion is that the large initial values on $2\times 10^{-1} \Delta t$ were left unoptimized after the training; the orange histogram in the second-left panel of Figure 4, representing the post-training distribution of $\Delta t$, shows a noticeable overlap with the random initialization (depicted by the blue histogram).


> Reading Figure 3, it seems that there is only a primacy effect, and no recency effect (in fact, the opposite).

Indeed, the emergence of the primacy effect (rather than the recency) is precisely our main finding.
If by "in fact, the opposite" you are referring to the theory predicting the recency, not the primacy, then that exactly highlights the novelty of our study: the observed primacy effect runs counter to theoretical expectations.


> The authors claim that intrinsic properties of the model architecture cause the primacy effect, rather than biases in the training data or task design, as the effect is also observed when the state and input matrices are not optimized for the task. This finding suggests that a more elaborate study of architectures and specific choices is needed, to determine how general this finding is and to what degree other (modern!) models exhibit the same bias.

Again, what we meant by "the observed primacy effect is attributed to the intrinsic properties of the SSM per se" was NOT intended as a comparison with alternative neural networks.
Rather, our claim is situated in contrast to prior studies on LLMs, which were trained on natural language data and thus can inherit human biases.
From this perspective, we believe that our experiment provides concrete evidence that the SSM can exhibit the primacy effect in the absence of human-originated biases because the synthetic training data were statistically balanced.

We would also want to clarify that the primacy effect is not necessarily a desired property of neural networks (from an engineering perspective), despite its potential usefulness in language modeling (cf. Xiao et al, 2024).
Thus, there is no reason to expect "modern models" to exhibit the memory bias, as they were not developed for reproducing cognitive phenomena.


> Clarity: The paper is generally well written but quite dense; it does seem to be reproducible given code. Figure 3 however is hard to interpret, specifically for how the primacy effect manifests from the graphs, and how general the findings are.


We have added a new paragraph in the beginning of the Results section.
We hope that it now clarifies the interpretation of the figure.
Specifically, the accuracy for all combinations of memorization and verification times is represented in the main square heatmaps.
The brightness of each cell in the square heatmaps indicates the accuracy for study items that were presented at the time indexed by the corresponding row (running top to bottom) and queried at the time indexed by the corresponding column (running left to right).
Therefore, the primacy effect is represented by the high accuracy (bright, yellow color) in the top rows of the heatmaps.