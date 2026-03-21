# openviking的差效果案例分析

## syllabus_dev(100QA)

### 检索失败 (Recall = 0)

这类问题是 **RAG 系统中最关键的瓶颈**，因为当包含答案的正确文档（Evidence）未能被检索系统召回并放入上下文时，大语言模型即便能力再强，也无法基于缺失的信息作出正确回答。检索失败通常表现为模型回复 “Not mentioned”，而实际上答案存在于知识库的某个角落。

#### 类型一：检索失败 · 深层文档被顶层淹没

**代表案例：Session 3 议题选择**

- **问题 (Question):** "What are the topics available to choose from for the session 3 presentation?"
- **上下文/证据位置 (Evidence):** 答案位于课程大纲较深层的目录 Course_Schedule_Each_Week_at_a_Glance/Session_3... 文件中，其中详细列出了第三次会议可选择的报告主题。
- **模型输出 (Model Output):** Not mentioned.
- **原因简析:** 用户的问题包含了“session”、“topics”等通用关键词，这些词在顶层的课程摘要（.abstract.md）和概览（.overview.md）等文档中频繁出现且得分很高。这导致检索系统优先返回了这些高层、概括性的文档，而真正包含具体答案的 **深层文档因为得分较低而被淹没**，未能进入模型的上下文。

**工程优化建议**

- **分层检索策略：** 明确 L0/L1（摘要/概览）与 L2（详细内容）文档的功能分工。让高层文档主要用于 **目录定位与语义导航**，而非直接提供答案。在最终生成答案时，应 **优先选择 L2 层的详细证据**。
- **控制高层摘要占比：** 在检索结果的候选集中，限制 .abstract.md 或 .overview.md 等摘要类文档的数量，例如“每个目录最多保留一个高层摘要”，以避免它们挤占包含具体信息的 L2 文档名额。
- **增加“具体性优先”偏置：** 在递归检索过程中，当检测到问题包含“Session 3”这类具体实体或编号时，**应优先向更深层的目录展开**，提高对具体问题的深层证据召回率。

#### **类型二：检索失败 · 证据碎片化与短块劣势**

**代表案例：学生成果要求**

- **问题 (Question):** "What student outcomes should be attained?"
- **上下文/证据位置 (Evidence):** 包含“student outcomes”的文档在切分过程中被拆分为多个小块（Chunks）。包含关键答案的那个 **chunk 非常短小**。
- **模型输出 (Model Output):** Not mentioned.
- **原因简析:** 包含答案的文本块因为过短，其语义表征不够丰富和稳定，导致在向量检索中得分较低。与此同时，文档中另一个 **更长但相关性较弱的 chunk 反而获得了更高的分数**，并被错误地召回。最终，正确的短证据块在竞争中失败，未能进入上下文。

**工程优化建议**

1. **优化文档切分策略：** 采用如 **滑动窗口（Sliding Window）** 的切分方法，确保一个完整的语义单元（如一个段落或一个列表）不会在切分边界被强制截断。
2. **保留结构化信息：** 在切分 chunk 时，为其附加上下文信息，如 **文档标题或章节路径** (Course_Requirements > Student_Outcomes)。这样，即使 chunk 本身很短，其 embedding 也能包含更丰富的结构化语义，增强表征能力。

#### **类型三：检索失败 · Embedding/Rerank 误杀**

**代表案例：Office Hour 参与方式**

- **问题 (Question):** "How can I attend an office hour?"
- **上下文/证据位置 (Evidence):** 答案与另一个相关问题 "What are the professors office hours?" 位于同一个文档 chunk 中。
- **模型输出 (Model Output):** Not mentioned.

- **原因简析:** 虽然正确答案所在的 chunk 在初步检索时可能已进入候选集（例如 Top 20），但由于 **Query Embedding 表征不佳** 或 **Rerank 模型排序问题**，它的得分过低，最终在精排阶段被过滤掉，未能进入最终的 Top-K 上下文。有趣的是，一个措辞更直接的相似问题（"What are the professors office hours?"）却能成功召回该 chunk。这表明问题的微小表述差异可能导致 embedding 空间的巨大距离。

**工程优化建议**

1. **统一使用 Retrieval Instruction：** 对于采用指令微调（Instruction-tuned）的 embedding 模型，应在计算查询向量前，统一添加指令前缀，例如 **"Find relevant passages for this question: {query}"**。这能激活模型的检索模式，使查询向量与文档向量更好地对齐，减少因问题表述差异导致的召回失败。
2. **调整 Rerank 参数：** 适当扩大初始候选集规模或降低 Rerank 阶段的过滤阈值。如果发现正确 chunk 频繁在 Rerank 后被丢弃，可以考虑 **增大最终传入上下文的 Top-K 数量**，以牺牲少量精度为代价换取更高的召回率。

### 生成错误 (Recall > 0)

在这类问题中，相关的证据已经成功进入了模型的上下文，但模型最终给出的答案却是错误的。这通常与模型的推理能力、对上下文的理解或信息提取的完整性有关。

#### **类型一：生成错误 · 推理型问题**

**代表案例：先修课程要求**

- **问题 (Question):** "Will I meet the prerequisites even though I haven't studied biochemistry?"
- **上下文/证据位置 (Evidence):** 文档中明确提到先修课程要求是 **“Organic chemistry is required and FS541-Food chemistry I is preferred.”**
- *模型输出 (Model Output):** Not mentioned.
- **标答 (Gold Answer):** Yes.
- **原因简析:** 文档中并未提及“biochemistry”，模型严格遵循“仅根据字面信息回答”的原则，因此判定问题无法回答。然而，正确的做法是进行一个简单的逻辑推理：既然要求里没有生物化学，那么没学过生物化学也满足要求。这暴露出当前系统的 Prompt **过度限制了模型的推理能力**。

**工程优化建议**

**设计问题类型感知的 Prompt：** 针对需要推理的问题（如 Y/N 判断、因果关系等），设计特定的 Prompt 模板。可以加入 **“You may perform logical reasoning based on the context, but do not use any external knowledge. Explain your reasoning briefly.”** 这样的指令，明确鼓励模型在上下文证据的基础上进行安全、有限的推理。

#### **类型二：生成错误 · 上下文干扰导致幻觉**

**代表案例：讨论课地点**

- **问题 (Question):** "Where will discussions be held?"
- **上下文/证据位置 (Evidence):** 文档中明确指出讨论课在 **"Moodle and in class"** 进行。
- **模型输出 (Model Output):** Moodle and Zoom.
- **原因简析:** 模型产生了 **“Zoom”** 这个幻觉，很可能是因为传入的上下文中包含了其他不相关但提及 “Zoom” 的片段，例如 **“Office hour on Zoom”**。模型未能准确区分不同场景下的地点信息，将不相关的上下文片段混淆在了一起。

**工程优化建议**

**引入“引用证据再回答”结构：** 在 Prompt 中要求模型采用两步式回答。第一步，**“Quote the relevant evidence from the context.”**；第二步，**“Answer the question based only on the quoted evidence.”** 这种结构强制模型聚焦于最直接相关的证据，可以有效提高答案的可解释性，并减少来自其他上下文片段的干扰。

#### **类型三：生成错误 · 信息提取不完整**

**代表案例：考试安排**

- **问题 (Question):** "Will you offer information about the exams and when they will be held?"
- **上下文/证据位置 (Evidence):** 文档中分散地列出了关于 Quiz 1-4, Midterm, Final Exam 的所有时间安排。
- **模型输出 (Model Output):** 仅列出了 Midterm 和 Final 的时间。
- **标答 (Gold Answer):** 完整列出所有 Quiz、Midterm 和 Final Exam 的时间。
- **原因简析:** 模型只返回了部分信息，遗漏了关于 Quiz 的部分。这表明虽然所有证据都在上下文中，但模型在进行信息汇总时未能 **完整地聚合所有相关要点**。这可能是因为 Prompt 中没有明确强调“完整性”的要求。

**工程优化建议**

**为汇总型问题设计专门 Prompt：** 当识别出问题是“汇总型”（例如包含 "what are", "list all" 等），使用特定的 Prompt，并明确强调完整性要求，例如 **“Summarize all relevant information from the context. Ensure that no important item is omitted.”**

#### **类型四：生成错误 · 问题语义与答案预期不匹配**

**代表案例：Quiz 频率**

- **问题 (Question):** "How often will there be quizzes?"
- **上下文/证据位置 (Evidence):** 文档中没有直接说明频率，但列出了 Quiz 1, Quiz 2, Quiz 3, Quiz 4 的具体日期。
- **模型输出 (Model Output):** Not mentioned.
- **标答 (Gold Answer):** Total 4.
- **原因简析:** 模型严格地在上下文中寻找匹配 “how often” (频率) 的直接表述（如 “weekly”, “every two weeks”），但没有找到。因此，它认为答案未提及。然而，用户的真实意图是想知道 Quiz 的数量。模型未能从对日期的列举中 **推断出总数为 4**，体现了对问题语义与用户预期的理解偏差。

**工程优化建议**

**增强 Prompt 的灵活性和推理引导：** 除了字面匹配，鼓励模型理解用户意图。可以加入类似 **“If a direct answer is not available, analyze the context to infer the most likely answer that satisfies the user's question.”** 的指令，引导模型在必要时进行信息转换和归纳，以更好地满足用户的查询预期。

### 评测问题 (Evaluation Misalignment)

部分被标记为 Bad Case 的例子，实际上并非 RAG 系统本身犯错，而是源于评测标准存在问题。这类问题虽然不直接指向工程优化，但对于准确评估系统性能至关重要。

#### **类型一：评测问题 · Bad Gold Answer**

**代表案例：额外费用**

- **问题 (Question):** "Are there any additional costs...?"
- **上下文/证据位置 (Evidence):** 文档提到需要 “textbook”，但紧接着说明 **“Available through the UofSC library”**。
- **模型输出 (Model Output):** No additional costs mentioned. (或类似表述)
- **标答 (Gold Answer):** Yes, textbook.
- **原因简析:** 这里的 **标答是错误的**。既然教科书可以通过图书馆免费获取，那么就不构成“额外费用”。模型根据上下文给出了正确的判断，但由于标答本身有问题，导致模型被误判为错误。

**优化建议**

**审查和修正数据集：** 在进行评测前，对 Gold Answer 进行一轮人工审查和清洗，修正那些与文档证据不符或逻辑错误的标答，确保评测基准的准确性。

#### **类型二：评测问题 · 常识推理 vs. 文档回答**

**代表案例：醉酒上课**

- **问题 (Question):** "Is it wise to come to class while intoxicated?"
- **上下文/证据位置 (Evidence):** Syllabus（课程大纲）中 **没有明确提到** 关于 “intoxicated”（醉酒）的任何规定。
- **模型输出 (Model Output):** Not mentioned.
- **标答 (Gold Answer):** No.
- **原因简析:** RAG 系统严格依据文档内容作出回答，是 **忠于原文** 的表现。而标答则引入了 **常识推理**：醉酒上课显然是不明智的。这造成了“基于文档的回答”与“基于常识的回答”之间的冲突。对于一个严格的 RAG 系统，其首要原则是依据给定文本，因此“Not mentioned”是更符合其设计目标的答案。

**优化建议**

**明确评测边界：** 在评测设计中，应明确区分 **“文本可直接回答”** 和 **“需要常识/世界知识”** 的问题。对于前者，应要求答案严格基于文档；对于后者，则需要评估模型在开放域下的知识和推理能力，或者将其从 RAG 评测中分离。

#### **类型三：评测问题 · 单位或粒度不一致**

**代表案例：学分与小时数**

- **问题 (Question):** "How many credits...?"
- **上下文/证据位置 (Evidence):** 文档中的原文是 **“Credit Hours: 4 hours”**。
- **模型输出 (Model Output):** 4 hours.
- **标答 (Gold Answer):** 4 credits.
- **原因简析:** 模型的回答 “4 hours” 是对原文的直接抽取，而标答 “4 credits” 则是根据问题进行的语义等价转换。虽然两者意思相同，但由于评测系统没有进行 **单位归一化**，导致模型被误判为错误。这属于 **答案粒度或表达形式不一致** 的问题。

**优化建议**

**增强评测鲁棒性：** 在自动化评测脚本中加入对 **等价表述的识别能力**，例如单位归一化（credits vs. hours）、同义词替换等。可以利用更强大的语言模型作为评测裁判（LLM-as-a-Judge），以更好地判断语义上的一致性，而不仅仅是字符串匹配。